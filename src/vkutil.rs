use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::fmt;
use std::sync::{Arc, Weak};

#[allow(unused_imports)]
use tracing::{debug, error, info, info_span, instrument, span, trace, warn, Level};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

#[derive(Debug)]
pub struct ComputePipeline {
    inner: vk::Pipeline,
}

impl ComputePipeline {
    #[instrument(name = "ComputePipeline::from_glsl", level = "info", err, skip(device))]
    pub fn from_glsl(
        device: ash::Device,
        pipeline_layout: vk::PipelineLayout,
        source_text: &str,
        source_filename: &str,
        entry_point: &str,
    ) -> Result<Self> {
        let mut compiler = shaderc::Compiler::new().ok_or("Failed to initialize spirv compiler")?;

        let artifact = compiler.compile_into_spirv(
            source_text,
            shaderc::ShaderKind::Compute,
            source_filename,
            entry_point,
            None,
        )?;

        Self::from_spv(device, pipeline_layout, artifact.as_binary(), entry_point)
    }

    #[instrument(name = "ComputePipeline::from_spv", level = "info", err, skip(device))]
    pub fn from_spv(
        device: ash::Device,
        pipeline_layout: vk::PipelineLayout,
        spv_binary: &[u32],
        entry_point: &str,
    ) -> Result<Self> {
        let entry_point_c_string = CString::new(entry_point)?;

        unsafe {
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(spv_binary),
                None,
            )?;

            let compile_result = device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::builder()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(entry_point_c_string.as_c_str())
                            .build(),
                    )
                    .layout(pipeline_layout)
                    .build()],
                None,
            );
            device.destroy_shader_module(module, None);

            match compile_result {
                Ok(compute_pipelines) => {
                    let compute_pipeline = ComputePipeline {
                        inner: compute_pipelines[0],
                    };

                    Ok(compute_pipeline)
                }
                Err(err) => Err(err.1.into()),
            }
        }
    }

    pub fn raw(&self) -> vk::Pipeline {
        self.inner
    }
}

/// Returns a vector of all desired instance extensions
fn select_instance_extensions(surface_extensions: Vec<CString>) -> Vec<CString> {
    let mut exts = Vec::new();

    // Add in all required surface extensions
    exts.extend(surface_extensions);

    // Add the debug utils extension
    let debug_utils_ext_name = CString::new(ext::DebugUtils::name().to_bytes()).unwrap();
    exts.push(debug_utils_ext_name);

    exts
}

/// Returns a vector of all desired instance layers
fn select_instance_layers(enable_validation: bool) -> Vec<CString> {
    let mut exts = Vec::new();

    // If the caller wants API validation, make sure we add the instance layer here
    if enable_validation {
        exts.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
    }

    exts
}

/// Returns a vector of all desired device extensions
fn select_device_extensions() -> Vec<CString> {
    let mut exts = Vec::new();

    // Add the swapchain extension
    let swapchain_ext_name = CString::new(khr::Swapchain::name().to_bytes()).unwrap();
    exts.push(swapchain_ext_name);

    exts
}

pub struct VkSurface {
    inner: vk::SurfaceKHR,
    ext: khr::Surface,
}

impl VkSurface {
    pub fn new(instance: &VkInstance, window: &winit::window::Window) -> Result<Self> {
        unsafe {
            // Create a surface from winit window.
            let surface =
                ash_window::create_surface(&instance.entry, &instance.inner, window, None)?;
            let ext = khr::Surface::new(&instance.entry, &instance.inner);

            Ok(Self {
                inner: surface,
                ext,
            })
        }
    }
}

impl Drop for VkSurface {
    fn drop(&mut self) {
        unsafe {
            self.ext.destroy_surface(self.inner, None);
        }
    }
}

impl fmt::Debug for VkSurface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VkSurface")
            // Not sure what we can display here?
            // Maybe available formats or something...?
            .finish()
    }
}

// Wrapper structure used to create and manage a Vulkan instance
pub struct VkInstance {
    inner: ash::Instance,
    entry: ash::Entry,
}

impl VkInstance {
    #[instrument(name = "VkInstance::new", level = "info", err, skip(window))]
    pub fn new(window: &winit::window::Window, enable_validation: bool) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::new()?;
            // Query all extensions required for swapchain usage
            let surface_extensions = ash_window::enumerate_required_extensions(window)?
                .iter()
                .map(|ext| CString::new(ext.to_bytes()).unwrap())
                .collect::<Vec<_>>();
            let instance_extension_strings = select_instance_extensions(surface_extensions);
            let instance_extensions = instance_extension_strings
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            let instance_layer_strings = select_instance_layers(enable_validation);
            let instance_layers = instance_layer_strings
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_version(1, 2, 0));
            let instance_desc = vk::InstanceCreateInfo::builder()
                .application_info(&app_desc)
                .enabled_extension_names(&instance_extensions)
                .enabled_layer_names(&instance_layers);

            let instance = entry.create_instance(&instance_desc, None)?;

            Ok(Self {
                inner: instance,
                entry,
            })
        }
    }

    pub fn raw(&self) -> &ash::Instance {
        &self.inner
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        unsafe {
            self.inner.destroy_instance(None);
        }
    }
}

impl fmt::Debug for VkInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VkInstance")
            .field("inner", &self.inner.handle())
            // Omit `self.entry` because there's nothing useful to display
            .finish()
    }
}

pub struct VkDebugMessenger {
    inner: vk::DebugUtilsMessengerEXT,
    debug_messenger_ext: ext::DebugUtils,
}

impl VkDebugMessenger {
    pub fn new(instance: &VkInstance) -> Result<Self> {
        unsafe {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_messenger_ext = ext::DebugUtils::new(&instance.entry, &instance.inner);
            let debug_messenger =
                debug_messenger_ext.create_debug_utils_messenger(&debug_info, None)?;

            Ok(Self {
                inner: debug_messenger,
                debug_messenger_ext,
            })
        }
    }
}

impl Drop for VkDebugMessenger {
    fn drop(&mut self) {
        unsafe {
            self.debug_messenger_ext
                .destroy_debug_utils_messenger(self.inner, None);
        }
    }
}

impl fmt::Debug for VkDebugMessenger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VkDebugMessenger")
            // Not sure what we can display here?
            .finish()
    }
}

/// Returns the index for the best queue family to use given the provided usage flags
fn find_best_queue_for_usage(
    queue_properties: &[vk::QueueFamilyProperties],
    usage: vk::QueueFlags,
) -> usize {
    let mut queue_index = usize::MAX;
    let mut min_support_bits = u32::MAX;
    for (idx, queue_properties) in queue_properties.iter().enumerate() {
        if queue_properties.queue_flags.contains(usage) {
            let support_bits = queue_properties.queue_flags.as_raw().count_ones();
            if support_bits < min_support_bits {
                min_support_bits = support_bits;
                queue_index = idx;
            }
        }
    }

    queue_index as usize
}

enum VkQueueType {
    Graphics = 0,
    Compute,
    Transfer,
}
const VK_QUEUE_TYPE_COUNT: usize = 3;

/// A wrapper around a Weak reference to an ash Device
///
/// This is wrapped to give better control over trait impls.
#[repr(transparent)]
#[derive(Clone)]
pub struct DeviceRef(pub Weak<ash::Device>);

impl DeviceRef {
    fn get(&self) -> Arc<ash::Device> {
        self.0
            .upgrade()
            .expect("Vulkan device destroyed while in use.")
    }
}

impl fmt::Debug for DeviceRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // If we don't have a handle, we have a bug!
        // Unlike most places, we want to report that and keep going.
        let handle: Option<ash::vk::Device> = self.0.upgrade().map(|d| d.handle());

        f.debug_struct("DeviceRef")
            .field("handle", &handle)
            .finish()
    }
}

/// A wrapper around a Weak reference to a vk_mem VkAllocator
///
/// This is wrapped to give better control over trait impls.
#[repr(transparent)]
#[derive(Clone)]
pub struct AllocRef(pub Weak<vk_mem::Allocator>);

impl AllocRef {
    fn get(&self) -> Arc<vk_mem::Allocator> {
        self.0
            .upgrade()
            .expect("Vulkan allocator destroyed while in use.")
    }
}

impl fmt::Debug for AllocRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        type VkMemResult<T> = std::result::Result<T, vk_mem::Error>;

        #[derive(Debug)]
        struct AllocRefInfo {
            physical_device_props: VkMemResult<vk::PhysicalDeviceProperties>,
            memory_props: VkMemResult<vk::PhysicalDeviceMemoryProperties>,
            vma_stats: VkMemResult<vk_mem::ffi::VmaStats>,
        }

        let info: Option<AllocRefInfo> = self.0.upgrade().map(|alloc| AllocRefInfo {
            physical_device_props: alloc.get_physical_device_properties(),
            memory_props: alloc.get_memory_properties(),
            vma_stats: alloc.calculate_stats(),
        });

        f.debug_struct("AllocRef").field("info", &info).finish()
    }
}

pub struct VkDevice {
    inner: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    queues_by_type: [vk::Queue; VK_QUEUE_TYPE_COUNT],
    queue_family_indices_by_type: [usize; VK_QUEUE_TYPE_COUNT],
    present_queue: vk::Queue,
}

impl VkDevice {
    pub fn new(
        instance: &VkInstance,
        physical_device: vk::PhysicalDevice,
        surface: &VkSurface,
    ) -> Result<Self> {
        unsafe {
            let device_extension_strings = select_device_extensions();
            let device_extensions = device_extension_strings
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();

            let queue_family_properties = instance
                .inner
                .get_physical_device_queue_family_properties(physical_device);

            // Identify a suitable queue family index for presentation
            let mut present_queue_family_index = u32::MAX;
            for idx in 0..queue_family_properties.len() {
                if surface.ext.get_physical_device_surface_support(
                    physical_device,
                    idx as u32,
                    surface.inner,
                )? {
                    present_queue_family_index = idx as u32;
                    break;
                }
            }

            // Initialize all available queue types
            let queue_infos = queue_family_properties
                .iter()
                .enumerate()
                .map(|(idx, _info)| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(idx as u32)
                        .queue_priorities(&[1.0])
                        .build()
                })
                .collect::<Vec<_>>();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extensions);

            let device =
                instance
                    .inner
                    .create_device(physical_device, &device_create_info, None)?;

            let queues = queue_infos
                .iter()
                .enumerate()
                .map(|(idx, _info)| device.get_device_queue(idx as u32, 0))
                .collect::<Vec<_>>();

            let queue_family_indices_by_type = [
                find_best_queue_for_usage(&queue_family_properties, vk::QueueFlags::GRAPHICS),
                find_best_queue_for_usage(&queue_family_properties, vk::QueueFlags::COMPUTE),
                find_best_queue_for_usage(&queue_family_properties, vk::QueueFlags::TRANSFER),
            ];

            let queues_by_type = [
                queues[queue_family_indices_by_type[0]],
                queues[queue_family_indices_by_type[1]],
                queues[queue_family_indices_by_type[2]],
            ];

            let present_queue = queues[present_queue_family_index as usize];

            Ok(Self {
                inner: Arc::new(device),
                physical_device,
                queues_by_type,
                queue_family_indices_by_type,
                present_queue,
            })
        }
    }

    pub fn raw(&self) -> Arc<ash::Device> {
        self.inner.clone()
    }

    pub fn as_ref(&self) -> DeviceRef {
        DeviceRef(Arc::downgrade(&self.inner))
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.queues_by_type[VkQueueType::Graphics as usize]
    }

    pub fn compute_queue(&self) -> vk::Queue {
        self.queues_by_type[VkQueueType::Compute as usize]
    }

    pub fn transfer_queue(&self) -> vk::Queue {
        self.queues_by_type[VkQueueType::Transfer as usize]
    }

    pub fn graphics_queue_family_index(&self) -> usize {
        self.queue_family_indices_by_type[VkQueueType::Graphics as usize]
    }

    pub fn compute_queue_family_index(&self) -> usize {
        self.queue_family_indices_by_type[VkQueueType::Compute as usize]
    }

    pub fn transfer_queue_family_index(&self) -> usize {
        self.queue_family_indices_by_type[VkQueueType::Transfer as usize]
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }
}

impl Drop for VkDevice {
    fn drop(&mut self) {
        unsafe {
            self.inner.destroy_device(None);
        }
    }
}

impl fmt::Debug for VkDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VkDevice")
            .field("inner", &self.inner.handle())
            .field("physical_device", &self.physical_device)
            .field("queues_by_type", &self.queues_by_type)
            .field(
                "queue_family_indices_by_type",
                &self.queue_family_indices_by_type,
            )
            .field("present_queue", &self.present_queue)
            .finish()
    }
}

pub struct VkSwapchain {
    inner: vk::SwapchainKHR,
    ext: ash::extensions::khr::Swapchain,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub images: Vec<vk::Image>,
}

impl VkSwapchain {
    pub fn new(
        instance: &VkInstance,
        surface: &VkSurface,
        device: &VkDevice,
        width: u32,
        height: u32,
        old_swapchain: Option<&VkSwapchain>,
    ) -> Result<Self> {
        unsafe {
            let surface_formats = surface
                .ext
                .get_physical_device_surface_formats(device.physical_device, surface.inner)?;
            let surface_format = if (surface_formats.len() == 1)
                && (surface_formats[0].format == vk::Format::UNDEFINED)
            {
                // Undefined means we get to choose our format
                vk::SurfaceFormatKHR::builder()
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .build()
            } else {
                // Attempt to select R8G8B8A8
                if let Some(format) = surface_formats
                    .iter()
                    .find(|surface| surface.format == vk::Format::R8G8B8A8_UNORM)
                {
                    *format
                // Fall back to B8R8G8A8
                } else if let Some(format) = surface_formats
                    .iter()
                    .find(|surface| surface.format == vk::Format::B8G8R8A8_UNORM)
                {
                    *format
                // If everything else fails, just use the first format in the list
                } else {
                    surface_formats[0]
                }
            };
            let surface_capabilities = surface
                .ext
                .get_physical_device_surface_capabilities(device.physical_device, surface.inner)?;
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D { width, height },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface
                .ext
                .get_physical_device_surface_present_modes(device.physical_device, surface.inner)?;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                // Prefer mailbox mode
                vk::PresentModeKHR::MAILBOX
            } else if present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                // Use fifo relaxed if mailbox isn't available
                vk::PresentModeKHR::FIFO_RELAXED
            } else {
                // Fall back to the required fifo mode if nothing else works
                vk::PresentModeKHR::FIFO
            };
            let ext = khr::Swapchain::new(&instance.inner, &*device.inner);

            let old_swapchain_handle = if let Some(old_swapchain) = old_swapchain {
                old_swapchain.inner
            } else {
                vk::SwapchainKHR::null()
            };

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface.inner)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .old_swapchain(old_swapchain_handle);

            let swapchain = ext.create_swapchain(&swapchain_create_info, None)?;

            let images = ext.get_swapchain_images(swapchain)?;

            Ok(Self {
                inner: swapchain,
                ext,
                surface_format,
                surface_resolution,
                images,
            })
        }
    }

    /// Attempts to acquire the next image in the swapchain
    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: Option<vk::Semaphore>,
        fence: Option<vk::Fence>,
    ) -> Result<(u32, bool)> {
        unsafe {
            Ok(self.ext.acquire_next_image(
                self.inner,
                timeout,
                semaphore.unwrap_or_default(),
                fence.unwrap_or_default(),
            )?)
        }
    }

    // Attempts to present the specified swapchain image on the display
    pub fn present_image(
        &self,
        index: u32,
        wait_semaphores: &[vk::Semaphore],
        queue: vk::Queue,
    ) -> Result<bool> {
        let swapchains = [self.inner];
        let image_indices = [index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { Ok(self.ext.queue_present(queue, &present_info)?) }
    }
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.ext.destroy_swapchain(self.inner, None);
        }
    }
}

impl fmt::Debug for VkSwapchain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("VkSwapchain")
            // Not sure what we can display here?
            .finish()
    }
}

#[derive(Debug)]
pub struct VkBuffer {
    inner: vk::Buffer,
    allocator: AllocRef,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
}

impl VkBuffer {
    pub fn new(
        allocator: AllocRef,
        buffer_info: &vk::BufferCreateInfo,
        allocation_create_info: &vk_mem::AllocationCreateInfo,
    ) -> Result<Self> {
        let (inner, allocation, allocation_info) = allocator
            .get()
            .create_buffer(buffer_info, allocation_create_info)?;
        Ok(VkBuffer {
            inner,
            allocator,
            allocation,
            allocation_info,
        })
    }

    pub fn raw(&self) -> vk::Buffer {
        self.inner
    }

    pub fn info(&self) -> &vk_mem::AllocationInfo {
        &self.allocation_info
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // TODO: This function always returns a successful result and should be modified to not
        //       return anything.
        self.allocator
            .get()
            .destroy_buffer(self.inner, &self.allocation)
            .unwrap();
    }
}

#[derive(Debug)]
pub struct VkImage {
    inner: vk::Image,
    allocator: AllocRef,
    allocation: vk_mem::Allocation,
}

impl VkImage {
    pub fn new(
        allocator: AllocRef,
        image_info: &vk::ImageCreateInfo,
        allocation_info: &vk_mem::AllocationCreateInfo,
    ) -> Result<Self> {
        let (inner, allocation, _alloc_info) =
            allocator.get().create_image(image_info, allocation_info)?;
        Ok(VkImage {
            inner,
            allocator,
            allocation,
        })
    }

    pub fn raw(&self) -> vk::Image {
        self.inner
    }
}

impl Drop for VkImage {
    fn drop(&mut self) {
        // TODO: This function always returns a successful result and should be modified to not
        //       return anything.
        self.allocator
            .get()
            .destroy_image(self.inner, &self.allocation)
            .unwrap();
    }
}

#[derive(Debug)]
pub struct VkImageView {
    inner: vk::ImageView,
    device: DeviceRef,
}

impl VkImageView {
    pub fn new(device: DeviceRef, create_info: &vk::ImageViewCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_image_view(create_info, None)? };
        Ok(VkImageView { inner, device })
    }

    pub fn raw(&self) -> vk::ImageView {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkImageView {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_image_view(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkSampler {
    inner: vk::Sampler,
    device: DeviceRef,
}

impl VkSampler {
    pub fn new(device: DeviceRef, create_info: &vk::SamplerCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_sampler(create_info, None)? };
        Ok(VkSampler { inner, device })
    }

    pub fn raw(&self) -> vk::Sampler {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkSampler {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_sampler(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkCommandPool {
    inner: vk::CommandPool,
    device: DeviceRef,
}

impl VkCommandPool {
    pub fn new(device: DeviceRef, create_info: &vk::CommandPoolCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_command_pool(create_info, None)? };
        Ok(VkCommandPool { inner, device })
    }

    pub fn raw(&self) -> vk::CommandPool {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }

    pub fn allocate_command_buffer(
        &self,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer> {
        let result = unsafe {
            self.device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.inner)
                    .level(level)
                    .command_buffer_count(1),
            )?
        };
        Ok(result[0])
    }

    pub fn free_command_buffer(&self, cmd_buffer: vk::CommandBuffer) {
        unsafe {
            self.device()
                .free_command_buffers(self.inner, &[cmd_buffer]);
        }
    }
}

impl Drop for VkCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_command_pool(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkSemaphore {
    inner: vk::Semaphore,
    device: DeviceRef,
}

impl VkSemaphore {
    pub fn new(device: DeviceRef, create_info: &vk::SemaphoreCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_semaphore(create_info, None)? };
        Ok(VkSemaphore { inner, device })
    }

    pub fn raw(&self) -> vk::Semaphore {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_semaphore(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkFence {
    inner: vk::Fence,
    device: DeviceRef,
}

impl VkFence {
    pub fn new(device: DeviceRef, create_info: &vk::FenceCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_fence(create_info, None)? };
        Ok(VkFence { inner, device })
    }

    pub fn raw(&self) -> vk::Fence {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkFence {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_fence(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkDescriptorSetLayout {
    inner: vk::DescriptorSetLayout,
    device: DeviceRef,
}

impl VkDescriptorSetLayout {
    pub fn new(device: DeviceRef, create_info: &vk::DescriptorSetLayoutCreateInfo) -> Result<Self> {
        let inner = unsafe {
            device
                .get()
                .create_descriptor_set_layout(create_info, None)?
        };
        Ok(VkDescriptorSetLayout { inner, device })
    }

    pub fn raw(&self) -> vk::DescriptorSetLayout {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device()
                .destroy_descriptor_set_layout(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkPipelineLayout {
    inner: vk::PipelineLayout,
    device: DeviceRef,
}

impl VkPipelineLayout {
    pub fn new(device: DeviceRef, create_info: &vk::PipelineLayoutCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_pipeline_layout(create_info, None)? };
        Ok(VkPipelineLayout { inner, device })
    }

    pub fn raw(&self) -> vk::PipelineLayout {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkPipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_pipeline_layout(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkDescriptorPool {
    inner: vk::DescriptorPool,
    device: DeviceRef,
}

impl VkDescriptorPool {
    pub fn new(device: DeviceRef, create_info: &vk::DescriptorPoolCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_descriptor_pool(create_info, None)? };
        Ok(VkDescriptorPool { inner, device })
    }

    pub fn raw(&self) -> vk::DescriptorPool {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }

    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let result = unsafe {
            self.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.inner)
                    .set_layouts(&[layout]),
            )?
        };
        Ok(result[0])
    }

    pub fn free_descriptor_set(&self, descriptor_set: vk::DescriptorSet) {
        unsafe {
            self.device()
                .free_descriptor_sets(self.inner, &[descriptor_set]);
        }
    }
}

impl Drop for VkDescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_descriptor_pool(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkShaderModule {
    inner: vk::ShaderModule,
    device: DeviceRef,
}

impl VkShaderModule {
    pub fn new(device: DeviceRef, create_info: &vk::ShaderModuleCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_shader_module(create_info, None)? };
        Ok(VkShaderModule { inner, device })
    }

    pub fn raw(&self) -> vk::ShaderModule {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_shader_module(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkPipeline {
    inner: vk::Pipeline,
    device: DeviceRef,
}

impl VkPipeline {
    fn from_pipeline(device: DeviceRef, inner: vk::Pipeline) -> Self {
        VkPipeline { inner, device }
    }

    pub fn raw(&self) -> vk::Pipeline {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_pipeline(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkPipelineCache {
    inner: vk::PipelineCache,
    device: DeviceRef,
}

impl VkPipelineCache {
    pub fn new(device: DeviceRef, create_info: &vk::PipelineCacheCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_pipeline_cache(create_info, None)? };
        Ok(VkPipelineCache { inner, device })
    }

    pub fn raw(&self) -> vk::PipelineCache {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }

    pub fn create_graphics_pipeline(
        &self,
        create_info: &vk::GraphicsPipelineCreateInfo,
    ) -> Result<VkPipeline> {
        let result = unsafe {
            self.device()
                .create_graphics_pipelines(self.inner, &[*create_info], None)
        };
        match result {
            Ok(pipelines) => Ok(VkPipeline::from_pipeline(self.device.clone(), pipelines[0])),
            Err((_pipelines, err)) => Err(err.into()),
        }
    }

    pub fn create_compute_pipeline(
        &self,
        create_info: &vk::ComputePipelineCreateInfo,
    ) -> Result<VkPipeline> {
        let result = unsafe {
            self.device()
                .create_compute_pipelines(self.inner, &[*create_info], None)
        };
        match result {
            Ok(pipelines) => Ok(VkPipeline::from_pipeline(self.device.clone(), pipelines[0])),
            Err((_pipelines, err)) => Err(err.into()),
        }
    }
}

impl Drop for VkPipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_pipeline_cache(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkRenderPass {
    inner: vk::RenderPass,
    device: DeviceRef,
}

impl VkRenderPass {
    pub fn new(device: DeviceRef, create_info: &vk::RenderPassCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_render_pass(create_info, None)? };
        Ok(VkRenderPass { inner, device })
    }

    pub fn raw(&self) -> vk::RenderPass {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_render_pass(self.inner, None);
        }
    }
}

#[derive(Debug)]
pub struct VkFramebuffer {
    inner: vk::Framebuffer,
    device: DeviceRef,
}

impl VkFramebuffer {
    pub fn new(device: DeviceRef, create_info: &vk::FramebufferCreateInfo) -> Result<Self> {
        let inner = unsafe { device.get().create_framebuffer(create_info, None)? };
        Ok(VkFramebuffer { inner, device })
    }

    pub fn raw(&self) -> vk::Framebuffer {
        self.inner
    }

    pub fn device(&self) -> Arc<ash::Device> {
        self.device.get()
    }
}

impl Drop for VkFramebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device().destroy_framebuffer(self.inner, None);
        }
    }
}
