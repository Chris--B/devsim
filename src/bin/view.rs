use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use imgui::{DrawCmd, DrawCmdParams};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use clap::Clap;
use devsim::vkutil::*;
use imgui::internal::RawWrapper;
use std::fmt;
use std::io;
use std::io::Write;
use std::slice;

#[allow(unused_imports)]
use tracing::{debug, error, info, info_span, instrument, span, trace, warn, Level};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Utility structure that simplifies the process of writing data that's constant over a single frame into GPU memory
struct ConstantDataWriter {
    buffer: *mut u8,
    buffer_size: usize,
    bytes_written: usize,
}

impl ConstantDataWriter {
    pub fn new(buffer: *mut u8, buffer_size: usize) -> Self {
        ConstantDataWriter {
            buffer,
            buffer_size,
            bytes_written: 0,
        }
    }

    pub fn dword_offset(&self) -> u32 {
        (self.bytes_written / 4) as u32
    }
}

impl io::Write for ConstantDataWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_remaining = self.buffer_size - self.bytes_written;
        let bytes_written = if buf.len() <= bytes_remaining {
            buf.len()
        } else {
            bytes_remaining
        };

        let buffer = unsafe {
            slice::from_raw_parts_mut(self.buffer.add(self.bytes_written), bytes_remaining)
        };
        buffer[..bytes_written].clone_from_slice(&buf[..bytes_written]);

        self.bytes_written += bytes_written;

        Ok(bytes_written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Selects a physical device from the provided list
fn select_physical_device(physical_devices: &[vk::PhysicalDevice]) -> vk::PhysicalDevice {
    // TODO: Support proper physical device selection
    //       For now, we just use the first device
    physical_devices[0]
}

/// Size of the scratch memory buffer in bytes that is available to each frame
const FRAME_MEMORY_SIZE: u64 = 8 * 1024 * 1024;

/// Number of individual texture slots available to shaders during a frame
const NUM_TEXTURE_SLOTS: u64 = 64;

/// Texture slot index associated with the imgui font
const IMGUI_FONT_TEXTURE_SLOT_INDEX: u64 = NUM_TEXTURE_SLOTS - 1;

#[derive(Debug)]
#[allow(dead_code)]
struct FrameState {
    fb_image_view: VkImageView,
    fb_image: VkImage,
    cmd_buffer: vk::CommandBuffer,
    fence: VkFence,
    descriptor_set: vk::DescriptorSet,
    rendering_finished_semaphore: VkSemaphore,
}

impl FrameState {
    #[instrument(name = "FrameState::new", level = "info", err)]
    fn new(
        device: &VkDevice,
        allocator: AllocRef,
        command_pool: &VkCommandPool,
        descriptor_pool: &VkDescriptorPool,
        descriptor_set_layout: &VkDescriptorSetLayout,
        frame_memory_buffer: &VkBuffer,
        fb_width: u32,
        fb_height: u32,
        imgui_renderer: &ImguiRenderer,
    ) -> Result<Self> {
        let cmd_buffer = command_pool.allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)?;

        let rendering_finished_semaphore =
            VkSemaphore::new(device.as_ref(), &vk::SemaphoreCreateInfo::default())?;
        let fence = VkFence::new(
            device.as_ref(),
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
        )?;

        let descriptor_set =
            descriptor_pool.allocate_descriptor_set(descriptor_set_layout.raw())?;

        let fb_image = VkImage::new(
            allocator,
            &ash::vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: fb_width,
                    height: fb_height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_UNORM)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;

        let fb_image_view = VkImageView::new(
            device.as_ref(),
            &vk::ImageViewCreateInfo::builder()
                .image(fb_image.raw())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY)
                        .build(),
                )
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                ),
        )?;
        unsafe {
            device.raw().update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(frame_memory_buffer.raw())
                        .offset(0)
                        .range(FRAME_MEMORY_SIZE)
                        .build()])
                    .build()],
                &[],
            );

            let mut image_infos = (0..(NUM_TEXTURE_SLOTS - 1))
                .map(|_| {
                    vk::DescriptorImageInfo::builder()
                        .image_view(fb_image_view.raw())
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build()
                })
                .collect::<Vec<_>>();
            image_infos.push(
                vk::DescriptorImageInfo::builder()
                    .image_view(imgui_renderer.font_atlas_image_view.raw())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build(),
            );
            device.raw().update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_infos)
                    .build()],
                &[],
            );
        }

        Ok(FrameState {
            fb_image_view,
            fb_image,
            cmd_buffer,
            fence,
            descriptor_set,
            rendering_finished_semaphore,
        })
    }
}

#[allow(dead_code)]
struct Renderer {
    imgui_renderer: ImguiRenderer,
    frame_states: Vec<FrameState>,
    fb_upload_buffer: VkBuffer,
    frame_memory_buffer: VkBuffer,
    image_available_semaphores: Vec<VkSemaphore>,
    framebuffers: Vec<VkFramebuffer>,
    renderpass: VkRenderPass,

    cmd_pool: VkCommandPool,

    sampler: VkSampler,
    pipeline_layout: VkPipelineLayout,

    descriptor_set_layout: VkDescriptorSetLayout,

    descriptor_pool: VkDescriptorPool,
    gfx_pipeline: VkPipeline,
    imgui_pipeline: VkPipeline,

    pipeline_cache: VkPipelineCache,
    cur_frame_idx: usize,
    cur_swapchain_idx: usize,
    swapchain_image_views: Vec<VkImageView>,
    swapchain: VkSwapchain,
    allocator: Arc<vk_mem::Allocator>,
    surface: VkSurface,
    device: VkDevice,
    _debug_messenger: Option<VkDebugMessenger>,
    instance: VkInstance,
}

impl fmt::Debug for Renderer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Renderer")
            // We can't derive Debug because the allocator field doesn't impl.
            // Anything that we want to display in a "debug fmt" of the Renderer
            // can go here.
            // ... not sure what to put here?
            .field("device", &self.device)
            .field("instance", &self.instance)
            .finish()
    }
}

impl Renderer {
    #[instrument(name = "Renderer::new", level = "info", err, skip(window))]
    fn new(
        window: &winit::window::Window,
        fb_width: u32,
        fb_height: u32,
        enable_validation: bool,
        context: &mut imgui::Context,
    ) -> Result<Self> {
        let instance = VkInstance::new(window, enable_validation)?;

        let _debug_messenger = if enable_validation {
            Some(VkDebugMessenger::new(&instance)?)
        } else {
            None
        };

        let physical_devices = unsafe { instance.raw().enumerate_physical_devices()? };
        let physical_device = select_physical_device(&physical_devices);

        let surface = VkSurface::new(&instance, window)?;

        let device = VkDevice::new(&instance, physical_device, &surface)?;

        let allocator = Arc::new(vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
            physical_device,
            device: (*device.raw()).clone(),
            instance: instance.raw().clone(),
            flags: vk_mem::AllocatorCreateFlags::NONE,
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        })?);
        let alloc_ref = AllocRef(Arc::downgrade(&allocator));

        let pipeline_cache =
            VkPipelineCache::new(device.as_ref(), &vk::PipelineCacheCreateInfo::default())?;

        let swapchain = VkSwapchain::new(
            &instance,
            &surface,
            &device,
            window.inner_size().width,
            window.inner_size().height,
            None,
        )?;

        let surface_format = swapchain.surface_format;
        let surface_resolution = swapchain.surface_resolution;
        let desired_image_count = swapchain.images.len() as u32;
        let queue_family_index = 0;

        let swapchain_image_views = swapchain
            .images
            .iter()
            .map(|image| {
                VkImageView::new(
                    device.as_ref(),
                    &vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        ),
                )
            })
            .collect::<Result<Vec<VkImageView>>>()?;

        let renderpass = VkRenderPass::new(
            device.as_ref(),
            &vk::RenderPassCreateInfo::builder()
                .attachments(&[vk::AttachmentDescription::builder()
                    .format(surface_format.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .build()])
                .subpasses(&[vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&[vk::AttachmentReference::builder()
                        .attachment(0)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .build()])
                    .build()]),
        )?;

        let framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                VkFramebuffer::new(
                    device.as_ref(),
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(renderpass.raw())
                        .attachments(&[image_view.raw()])
                        .width(surface_resolution.width)
                        .height(surface_resolution.height)
                        .layers(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let cmd_pool = VkCommandPool::new(
            device.as_ref(),
            &vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
        )?;

        let sampler = VkSampler::new(
            device.as_ref(),
            &vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(10000.0)
                .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK),
        )?;

        let descriptor_set_layout = VkDescriptorSetLayout::new(
            device.as_ref(),
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::COMPUTE,
                    )
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&[sampler.raw()])
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(NUM_TEXTURE_SLOTS as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ]),
        )?;

        let pipeline_layout = VkPipelineLayout::new(
            device.as_ref(),
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[descriptor_set_layout.raw()])
                .push_constant_ranges(&[vk::PushConstantRange::builder()
                    .offset(0)
                    .size((4 * std::mem::size_of::<u32>()) as u32)
                    .stage_flags(
                        vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::COMPUTE,
                    )
                    .build()]),
        )?;

        let descriptor_pool = VkDescriptorPool::new(
            device.as_ref(),
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(desired_image_count)
                .pool_sizes(&[
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .descriptor_count(desired_image_count)
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(desired_image_count)
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(desired_image_count * (NUM_TEXTURE_SLOTS as u32))
                        .build(),
                ]),
        )?;

        let pipelines_init = info_span!("Shaders Init");
        let pipelines_init_enter = pipelines_init.enter();

        let mut compiler = shaderc::Compiler::new().expect("Failed to create compiler");
        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        let shader_dir = std::env::current_dir().unwrap().join("src/shaders");
        compile_options.set_include_callback(move |name, _inc_type, _parent_name, _depth| {
            let path = shader_dir.join(name);
            if let Ok(content) = std::fs::read_to_string(&path) {
                Ok(shaderc::ResolvedInclude {
                    resolved_name: String::from(name),
                    content,
                })
            } else {
                Err(format!(
                    "Failed to load included shader code from {}.",
                    name
                ))
            }
        });

        let util_pipelines = info_span!("Loading Utility Pipelines");
        let util_pipelines_enter = util_pipelines.enter();

        let vert_source = include_str!("../shaders/FullscreenPass.vert");
        let frag_source = include_str!("../shaders/CopyTexture.frag");

        let vert_result = compiler.compile_into_spirv(
            vert_source,
            shaderc::ShaderKind::Vertex,
            "FullscreenPass.vert",
            "main",
            Some(&compile_options),
        )?;

        let vert_module = VkShaderModule::new(
            device.as_ref(),
            &vk::ShaderModuleCreateInfo::builder().code(vert_result.as_binary()),
        )?;

        let frag_result = compiler.compile_into_spirv(
            frag_source,
            shaderc::ShaderKind::Fragment,
            "CopyTexture.frag",
            "main",
            Some(&compile_options),
        )?;

        let frag_module = VkShaderModule::new(
            device.as_ref(),
            &vk::ShaderModuleCreateInfo::builder().code(frag_result.as_binary()),
        )?;

        let entry_point_c_string = std::ffi::CString::new("main").unwrap();
        let gfx_pipeline = pipeline_cache.create_graphics_pipeline(
            &vk::GraphicsPipelineCreateInfo::builder()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(vert_module.raw())
                        .name(entry_point_c_string.as_c_str())
                        .build(),
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(frag_module.raw())
                        .name(entry_point_c_string.as_c_str())
                        .build(),
                ])
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::builder().build())
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::BACK)
                        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                        .line_width(1.0),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                // Don't need depth state
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                        vk::PipelineColorBlendAttachmentState::builder()
                            .color_write_mask(
                                vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B
                                    | vk::ColorComponentFlags::A,
                            )
                            .build(),
                    ]),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .layout(pipeline_layout.raw())
                .render_pass(renderpass.raw())
                .subpass(0),
        )?;

        drop(util_pipelines_enter);
        let imgui_pipelines = info_span!("Loading ImGUI Pipelines");
        let imgui_pipelines_enter = imgui_pipelines.enter();

        let imgui_vert_source = include_str!("../shaders/ImguiTriangle.vert");
        let imgui_frag_source = include_str!("../shaders/ImguiTriangle.frag");

        let imgui_vert_result = compiler.compile_into_spirv(
            imgui_vert_source,
            shaderc::ShaderKind::Vertex,
            "ImguiTriangle.vert",
            "main",
            Some(&compile_options),
        )?;

        let imgui_vert_module = VkShaderModule::new(
            device.as_ref(),
            &vk::ShaderModuleCreateInfo::builder().code(imgui_vert_result.as_binary()),
        )?;

        let imgui_frag_result = compiler.compile_into_spirv(
            imgui_frag_source,
            shaderc::ShaderKind::Fragment,
            "ImguiTriangle.frag",
            "main",
            Some(&compile_options),
        )?;

        let imgui_frag_module = VkShaderModule::new(
            device.as_ref(),
            &vk::ShaderModuleCreateInfo::builder().code(imgui_frag_result.as_binary()),
        )?;

        let imgui_entry_point_c_string = std::ffi::CString::new("main").unwrap();
        let imgui_pipeline = pipeline_cache.create_graphics_pipeline(
            &vk::GraphicsPipelineCreateInfo::builder()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(imgui_vert_module.raw())
                        .name(imgui_entry_point_c_string.as_c_str())
                        .build(),
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(imgui_frag_module.raw())
                        .name(imgui_entry_point_c_string.as_c_str())
                        .build(),
                ])
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_binding_descriptions(&[vk::VertexInputBindingDescription::builder(
                        )
                        .binding(0)
                        .stride(std::mem::size_of::<imgui::DrawVert>() as u32)
                        .input_rate(vk::VertexInputRate::VERTEX)
                        .build()])
                        .vertex_attribute_descriptions(&[
                            vk::VertexInputAttributeDescription::builder()
                                .location(0)
                                .binding(0)
                                .format(vk::Format::R32G32_SFLOAT)
                                .offset(0)
                                .build(),
                            vk::VertexInputAttributeDescription::builder()
                                .location(1)
                                .binding(0)
                                .format(vk::Format::R32G32_SFLOAT)
                                .offset(8)
                                .build(),
                            vk::VertexInputAttributeDescription::builder()
                                .location(2)
                                .binding(0)
                                .format(vk::Format::R32_UINT)
                                .offset(16)
                                .build(),
                        ]),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewports(&[vk::Viewport::default()])
                        .scissors(&[vk::Rect2D::default()]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .polygon_mode(vk::PolygonMode::FILL)
                        .cull_mode(vk::CullModeFlags::NONE)
                        .line_width(1.0),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                // Don't need depth state
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                        vk::PipelineColorBlendAttachmentState::builder()
                            .color_write_mask(
                                vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B
                                    | vk::ColorComponentFlags::A,
                            )
                            .blend_enable(true)
                            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .color_blend_op(vk::BlendOp::ADD)
                            .src_alpha_blend_factor(vk::BlendFactor::ONE)
                            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                            .alpha_blend_op(vk::BlendOp::ADD)
                            .build(),
                    ]),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .layout(pipeline_layout.raw())
                .render_pass(renderpass.raw())
                .subpass(0),
        )?;

        drop(imgui_pipelines_enter);
        drop(pipelines_init_enter);

        let image_available_semaphores = swapchain
            .images
            .iter()
            .map(|_| VkSemaphore::new(device.as_ref(), &vk::SemaphoreCreateInfo::default()))
            .collect::<Result<Vec<_>>>()?;

        let image_size_bytes = fb_width * fb_height * 4;

        let fb_upload_buffer = VkBuffer::new(
            alloc_ref.clone(),
            &ash::vk::BufferCreateInfo::builder()
                .size((((image_size_bytes + 255) & !255) * desired_image_count) as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuOnly,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
        )?;

        let frame_memory_buffer = VkBuffer::new(
            alloc_ref.clone(),
            &ash::vk::BufferCreateInfo::builder()
                .size(FRAME_MEMORY_SIZE * (desired_image_count as u64))
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
        )?;

        let imgui_renderer = ImguiRenderer::new(&device, alloc_ref.clone(), context)?;

        let frame_states = swapchain_image_views
            .iter()
            .map(|_image_view| {
                FrameState::new(
                    &device,
                    alloc_ref.clone(),
                    &cmd_pool,
                    &descriptor_pool,
                    &descriptor_set_layout,
                    &frame_memory_buffer,
                    fb_width,
                    fb_height,
                    &imgui_renderer,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Renderer {
            imgui_renderer,
            frame_states,
            frame_memory_buffer,
            fb_upload_buffer,
            framebuffers,
            image_available_semaphores,
            renderpass,
            cmd_pool,
            sampler,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            gfx_pipeline,
            imgui_pipeline,
            pipeline_cache,
            cur_frame_idx: 0,
            cur_swapchain_idx: 0,
            swapchain_image_views,
            swapchain,
            allocator,
            surface,
            device,
            _debug_messenger,
            instance,
        })
    }

    fn get_cur_frame_state(&self) -> &FrameState {
        &self.frame_states[self.cur_swapchain_idx]
    }

    #[instrument(
        name = "Renderer::recreate_swapchain",
        level = "info",
        err,
        skip(window)
    )]
    fn recreate_swapchain(&mut self, window: &winit::window::Window) -> Result<()> {
        println!(
            "Recreating {}x{} swapchain!",
            window.inner_size().width,
            window.inner_size().height
        );

        // Make sure all previous rendering work is completed before we destroy the old swapchain resources
        self.wait_for_idle();

        let swapchain = VkSwapchain::new(
            &self.instance,
            &self.surface,
            &self.device,
            window.inner_size().width,
            window.inner_size().height,
            Some(&self.swapchain),
        )?;

        let surface_format = swapchain.surface_format;
        let surface_resolution = swapchain.surface_resolution;

        let swapchain_image_views = swapchain
            .images
            .iter()
            .map(|image| {
                VkImageView::new(
                    self.device.as_ref(),
                    &vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        ),
                )
            })
            .collect::<Result<Vec<VkImageView>>>()?;

        let framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                VkFramebuffer::new(
                    self.device.as_ref(),
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(self.renderpass.raw())
                        .attachments(&[image_view.raw()])
                        .width(surface_resolution.width)
                        .height(surface_resolution.height)
                        .layers(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.swapchain_image_views = swapchain_image_views;
        self.framebuffers = framebuffers;
        self.swapchain = swapchain;

        Ok(())
    }

    fn begin_frame(&mut self) -> vk::CommandBuffer {
        unsafe {
            // Acquire the current swapchain image index
            // TODO: Handle suboptimal swapchains
            let (image_index, _is_suboptimal) = self
                .swapchain
                .acquire_next_image(
                    u64::MAX,
                    Some(self.image_available_semaphores[self.cur_frame_idx].raw()),
                    None,
                )
                .unwrap();
            // TODO: This should never happen since we're already handling window resize events, but this could be handled
            // more robustly in the future.
            assert!(!_is_suboptimal);
            self.cur_swapchain_idx = image_index as usize;

            let frame_state = self.get_cur_frame_state();

            // Wait for the resources for this frame to become available
            self.device
                .raw()
                .wait_for_fences(&[frame_state.fence.raw()], true, u64::MAX)
                .unwrap();

            let cmd_buffer = frame_state.cmd_buffer;

            self.device
                .raw()
                .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            cmd_buffer
        }
    }

    fn begin_render(&self) {
        let frame_state = self.get_cur_frame_state();
        let framebuffer = &self.framebuffers[self.cur_swapchain_idx];
        unsafe {
            self.device.raw().cmd_begin_render_pass(
                frame_state.cmd_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.renderpass.raw())
                    .framebuffer(framebuffer.raw())
                    .render_area(
                        vk::Rect2D::builder()
                            .extent(self.swapchain.surface_resolution)
                            .build(),
                    )
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }]),
                vk::SubpassContents::INLINE,
            );
        }
    }

    fn end_render(&self) {
        let frame_state = self.get_cur_frame_state();
        unsafe {
            self.device
                .raw()
                .cmd_end_render_pass(frame_state.cmd_buffer);
        }
    }

    fn end_frame(&mut self, cmd_buffer: vk::CommandBuffer) {
        let frame_state = self.get_cur_frame_state();
        unsafe {
            self.device.raw().end_command_buffer(cmd_buffer).unwrap();

            // The user should always pass the same cmdbuffer back to us after a frame
            assert_eq!(frame_state.cmd_buffer, cmd_buffer);

            let wait_semaphores = [self.image_available_semaphores[self.cur_frame_idx].raw()];
            let command_buffers = [cmd_buffer];
            let signal_semaphores = [frame_state.rendering_finished_semaphore.raw()];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TOP_OF_PIPE])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();

            let fence = &frame_state.fence;
            self.device.raw().reset_fences(&[fence.raw()]).unwrap();
            self.device
                .raw()
                .queue_submit(self.device.present_queue(), &[submit_info], fence.raw())
                .unwrap();

            let _is_suboptimal = self
                .swapchain
                .present_image(
                    self.cur_swapchain_idx as u32,
                    &signal_semaphores,
                    self.device.present_queue(),
                )
                .unwrap();
            // TODO: This should never happen since we're already handling window resize events, but this could be handled
            // more robustly in the future.
            assert!(!_is_suboptimal);

            self.cur_frame_idx = (self.cur_frame_idx + 1) % self.swapchain.images.len();
        }
    }

    #[instrument(name = "Renderer::wait_for_idle", level = "info")]
    fn wait_for_idle(&self) {
        unsafe { self.get_device().device_wait_idle().unwrap() };
    }

    fn get_device(&self) -> Arc<ash::Device> {
        self.device.raw()
    }

    fn get_allocator(&self) -> AllocRef {
        AllocRef(Arc::downgrade(&self.allocator))
    }

    fn get_cur_swapchain_idx(&self) -> usize {
        self.cur_swapchain_idx
    }
    fn get_num_swapchain_images(&self) -> usize {
        self.swapchain.images.len()
    }
}

#[derive(Debug)]
struct ImguiRenderer {
    #[allow(dead_code)]
    font_atlas_image: VkImage,
    font_atlas_image_view: VkImageView,
}

impl ImguiRenderer {
    #[instrument(name = "ImguiRenderer::new", level = "info", err)]
    fn new(device: &VkDevice, allocator: AllocRef, context: &mut imgui::Context) -> Result<Self> {
        let font_atlas_image;
        let font_atlas_image_view;
        {
            let mut context_fonts = context.fonts();
            let font_atlas = context_fonts.build_alpha8_texture();

            font_atlas_image = VkImage::new(
                allocator.clone(),
                &ash::vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: font_atlas.width,
                        height: font_atlas.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .format(vk::Format::R8_UNORM)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::GpuOnly,
                    ..Default::default()
                },
            )?;
            font_atlas_image_view = VkImageView::new(
                device.as_ref(),
                &vk::ImageViewCreateInfo::builder()
                    .image(font_atlas_image.raw())
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .components(
                        vk::ComponentMapping::builder()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY)
                            .build(),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    ),
            )?;

            let cmd_pool = VkCommandPool::new(
                device.as_ref(),
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(device.graphics_queue_family_index() as u32),
            )?;
            let cmd_buffer = cmd_pool.allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)?;
            unsafe {
                let raw_device = device.raw();
                raw_device.begin_command_buffer(
                    cmd_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )?;

                // TODO: It would be faster to upload this with the transfer queue, but it would significantly increase
                //       the complexity of the upload process here. Replace this with a more standardized resource
                //       upload process when it becomes available.
                raw_device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(font_atlas_image.raw())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                let atlas_buffer_size =
                    ((font_atlas.width * font_atlas.height) as usize) * std::mem::size_of::<u8>();
                let atlas_buffer = VkBuffer::new(
                    allocator,
                    &ash::vk::BufferCreateInfo::builder()
                        .size(atlas_buffer_size as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::CpuToGpu,
                        flags: vk_mem::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    },
                )?;

                let atlas_data_src = font_atlas.data.as_ptr();
                let atlas_data_dst = atlas_buffer.info().get_mapped_data();
                core::ptr::copy_nonoverlapping(atlas_data_src, atlas_data_dst, atlas_buffer_size);

                raw_device.cmd_copy_buffer_to_image(
                    cmd_buffer,
                    atlas_buffer.raw(),
                    font_atlas_image.raw(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::builder()
                        .buffer_offset(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: font_atlas.width,
                            height: font_atlas.height,
                            depth: 1,
                        })
                        .build()],
                );

                raw_device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(font_atlas_image.raw())
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        )
                        .build()],
                );

                raw_device.end_command_buffer(cmd_buffer)?;
                raw_device.queue_submit(
                    device.graphics_queue(),
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[cmd_buffer])
                        .build()],
                    vk::Fence::null(),
                )?;
                raw_device.queue_wait_idle(device.graphics_queue())?;
            }
        }

        context.fonts().tex_id = imgui::TextureId::from(IMGUI_FONT_TEXTURE_SLOT_INDEX as usize);

        Ok(ImguiRenderer {
            font_atlas_image,
            font_atlas_image_view,
        })
    }
}

fn show(opts: &SimOptions) -> ! {
    let hw_init = info_span!("Hardware Init");
    let hw_init_enter = hw_init.enter();

    let mut hw_device = devsim::device::Device::new();
    hw_device
        .load_elf(&opts.elf_path)
        .expect("Failed to load elf file");

    drop(hw_init_enter);
    let window_init = info_span!("Window Init");
    let window_init_enter = window_init.enter();

    let (fb_width, fb_height) = hw_device
        .query_framebuffer_size()
        .expect("Failed to query framebuffer size");
    let image_size_bytes = fb_width * fb_height * 4;

    let window_width = 1280;
    let window_height = 720;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("DevSim View")
        .with_inner_size(winit::dpi::PhysicalSize::new(window_width, window_height))
        .build(&event_loop)
        .expect("Failed to create window");

    drop(window_init_enter);
    let imgui_init = info_span!("ImGUI Init");
    let imgui_init_enter = imgui_init.enter();

    let mut context = imgui::Context::create();
    context.set_renderer_name(Some(imgui::ImString::from(String::from("DevSim"))));
    context
        .io_mut()
        .backend_flags
        .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);

    let mut platform = WinitPlatform::init(&mut context);
    platform.attach_window(context.io_mut(), &window, HiDpiMode::Default);

    drop(imgui_init_enter);
    let gfx_init = info_span!("Gfx Init");
    let gfx_init_enter = gfx_init.enter();

    let mut renderer = Renderer::new(&window, fb_width, fb_height, true, &mut context)
        .expect("Failed to create renderer");

    // TODO: Find a better way to initialize these vectors
    // Note: We don't have (or want?) Clone on VkBuffer, so this doesn't work:
    //      vec![None; renderer.get_num_swapchain_images()]
    let mut imgui_vtx_buffers = Vec::new();
    for _i in 0..renderer.get_num_swapchain_images() {
        imgui_vtx_buffers.push(None);
    }
    let mut imgui_idx_buffers = Vec::new();
    for _i in 0..renderer.get_num_swapchain_images() {
        imgui_idx_buffers.push(None);
    }

    drop(gfx_init_enter);

    let mut last_frame = Instant::now();
    let mut frame_idx = 0;
    unsafe {
        event_loop.run(move |event, _, control_flow| {
            let event_loop = info_span!("Event Loop Iter", ?frame_idx,);
            let _event_loop_enter = event_loop.enter();

            platform.handle_event(context.io_mut(), &window, &event);

            match event {
                Event::NewEvents(StartCause::Init) => {
                    *control_flow = ControlFlow::Poll;
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(_new_size) => {
                        // TODO: This code needs to be updated to properly handle minimized windows
                        //       When a window is minimized, it resizes to 0x0 which causes all sorts of problems
                        //       inside the graphics api. This basically results in crashes on minimize. :/
                        //       This will be fixed in a future change.
                        renderer.recreate_swapchain(&window).unwrap();
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    let main_events = info_span!("main events cleared");
                    let _main_events_enter = main_events.enter();

                    let begin_frame = info_span!("Do Renderer `frame()`");
                    let begin_frame_enter = begin_frame.enter();
                    let cmd_buffer = renderer.begin_frame();

                    let now = Instant::now();
                    context.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                    frame_idx += 1;

                    platform
                        .prepare_frame(context.io_mut(), &window)
                        .expect("Failed to prepare frame");

                    let ui = context.frame();
                    // application-specific rendering *under the UI*

                    let hw_device_loop = info_span!("spin hw device");
                    let hw_device_loop_enter = hw_device_loop.enter();

                    // Enable the device
                    hw_device.enable();

                    loop {
                        match hw_device.query_is_halted() {
                            Ok(is_halted) => {
                                if !is_halted {
                                    // Still executing...
                                } else {
                                    break;
                                }
                            }
                            Err(err) => {
                                println!("Device error: {}", err);
                                break;
                            }
                        }
                    }
                    drop(hw_device_loop_enter);

                    let fb_upload_buffer = &renderer.fb_upload_buffer;
                    let p_fb_upload_buf_mem = fb_upload_buffer.info().get_mapped_data();
                    let p_current_fb_upload_buf_mem = p_fb_upload_buf_mem.offset(
                        (image_size_bytes * (renderer.get_cur_swapchain_idx() as u32)) as isize,
                    ) as *mut u8;
                    let mut current_fb_upload_buf_slice = core::slice::from_raw_parts_mut(
                        p_current_fb_upload_buf_mem,
                        image_size_bytes as usize,
                    );

                    hw_device
                        .dump_framebuffer(&mut current_fb_upload_buf_slice)
                        .expect("Failed to dump device framebuffer!");

                    let device = renderer.get_device();

                    let cur_fb_image = &renderer.get_cur_frame_state().fb_image;

                    // Initialize the current framebuffer image
                    device.cmd_pipeline_barrier(
                        cmd_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::empty())
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(cur_fb_image.raw())
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            )
                            .build()],
                    );

                    // Copy the latest device output to the framebuffer image
                    let buffer_offset =
                        (renderer.get_cur_swapchain_idx() as u32) * image_size_bytes;
                    device.cmd_copy_buffer_to_image(
                        cmd_buffer,
                        fb_upload_buffer.raw(),
                        cur_fb_image.raw(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::builder()
                            .buffer_offset(buffer_offset as u64)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_extent(vk::Extent3D {
                                width: fb_width,
                                height: fb_height,
                                depth: 1,
                            })
                            .build()],
                    );

                    // Make sure the fb image is ready to be read by shaders
                    device.cmd_pipeline_barrier(
                        cmd_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(cur_fb_image.raw())
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            )
                            .build()],
                    );

                    let frame_state = renderer.get_cur_frame_state();
                    let descriptor_set = frame_state.descriptor_set;

                    let begin_render = info_span!("Do Renderer `render()`");
                    let begin_render_enter = begin_render.enter();
                    renderer.begin_render();

                    device.cmd_bind_pipeline(
                        cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        renderer.gfx_pipeline.raw(),
                    );

                    device.cmd_set_viewport(
                        cmd_buffer,
                        0,
                        &[vk::Viewport::builder()
                            .x(0.0)
                            .y(0.0)
                            .width(window.inner_size().width as f32)
                            .height(window.inner_size().height as f32)
                            .build()],
                    );

                    device.cmd_set_scissor(
                        cmd_buffer,
                        0,
                        &[vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(
                                vk::Extent2D::builder()
                                    .width(window.inner_size().width)
                                    .height(window.inner_size().height)
                                    .build(),
                            )
                            .build()],
                    );

                    let constant_data_offset =
                        renderer.get_cur_swapchain_idx() * (FRAME_MEMORY_SIZE as usize);

                    device.cmd_bind_descriptor_sets(
                        cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        renderer.pipeline_layout.raw(),
                        0,
                        &[descriptor_set],
                        &[constant_data_offset as u32],
                    );

                    let mut constant_writer = ConstantDataWriter::new(
                        renderer
                            .frame_memory_buffer
                            .info()
                            .get_mapped_data()
                            .add(constant_data_offset),
                        FRAME_MEMORY_SIZE as usize,
                    );

                    device.cmd_draw(cmd_buffer, 3, 1, 0, 0);

                    // Show an ImGUI demo window for now
                    let mut opened = true;
                    ui.show_demo_window(&mut opened);

                    platform.prepare_render(&ui, &window);
                    let draw_data = ui.render();

                    let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
                    let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
                    if (fb_width > 0.0) && (fb_height > 0.0) && draw_data.total_idx_count > 0 {
                        let total_vtx_count = draw_data.total_vtx_count as usize;
                        let vtx_buffer_size =
                            total_vtx_count * std::mem::size_of::<imgui::DrawVert>();
                        let vtx_buffer = VkBuffer::new(
                            renderer.get_allocator(),
                            &ash::vk::BufferCreateInfo::builder()
                                .size(vtx_buffer_size as u64)
                                .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::CpuToGpu,
                                flags: vk_mem::AllocationCreateFlags::MAPPED,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        let vtx_buffer_slice = slice::from_raw_parts_mut(
                            vtx_buffer.info().get_mapped_data(),
                            vtx_buffer_size,
                        );
                        let vtx_buffer_raw = vtx_buffer.raw();

                        let total_idx_count = draw_data.total_idx_count as usize;
                        let idx_buffer_size =
                            total_idx_count * std::mem::size_of::<imgui::DrawIdx>();
                        let idx_buffer = VkBuffer::new(
                            renderer.get_allocator(),
                            &ash::vk::BufferCreateInfo::builder()
                                .size(idx_buffer_size as u64)
                                .usage(vk::BufferUsageFlags::INDEX_BUFFER),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::CpuToGpu,
                                flags: vk_mem::AllocationCreateFlags::MAPPED,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        let idx_buffer_slice = slice::from_raw_parts_mut(
                            idx_buffer.info().get_mapped_data(),
                            idx_buffer_size,
                        );
                        let idx_buffer_raw = idx_buffer.raw();

                        let mut vtx_bytes_written: usize = 0;
                        let mut vtx_buffer_offsets = Vec::new();

                        let mut idx_bytes_written: usize = 0;
                        let mut idx_buffer_offsets = Vec::new();

                        for draw_list in draw_data.draw_lists() {
                            let vtx_data_src = draw_list.vtx_buffer().as_ptr() as *const u8;
                            let vtx_data_dst =
                                (vtx_buffer_slice.as_mut_ptr() as *mut u8).add(vtx_bytes_written);
                            let vtx_data_size = draw_list.vtx_buffer().len()
                                * std::mem::size_of::<imgui::DrawVert>();
                            core::ptr::copy_nonoverlapping(
                                vtx_data_src,
                                vtx_data_dst,
                                vtx_data_size,
                            );
                            vtx_buffer_offsets.push(vtx_bytes_written);
                            vtx_bytes_written += vtx_data_size;

                            let idx_data_src = draw_list.idx_buffer().as_ptr() as *const u8;
                            let idx_data_dst =
                                (idx_buffer_slice.as_mut_ptr() as *mut u8).add(idx_bytes_written);
                            let idx_data_size = draw_list.idx_buffer().len()
                                * std::mem::size_of::<imgui::DrawIdx>();
                            core::ptr::copy_nonoverlapping(
                                idx_data_src,
                                idx_data_dst,
                                idx_data_size,
                            );
                            idx_buffer_offsets.push(idx_bytes_written);
                            idx_bytes_written += idx_data_size;
                        }

                        imgui_vtx_buffers[renderer.get_cur_swapchain_idx()] = Some(vtx_buffer);
                        imgui_idx_buffers[renderer.get_cur_swapchain_idx()] = Some(idx_buffer);

                        device.cmd_bind_pipeline(
                            cmd_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            renderer.imgui_pipeline.raw(),
                        );

                        let fb_scale = draw_data.framebuffer_scale;
                        device.cmd_set_viewport(
                            cmd_buffer,
                            0,
                            &[vk::Viewport::builder()
                                .x(draw_data.display_pos[0] * fb_scale[0])
                                .y(draw_data.display_pos[1] * fb_scale[1])
                                .width(draw_data.display_size[0] * fb_scale[0])
                                .height(draw_data.display_size[1] * fb_scale[1])
                                .build()],
                        );

                        let clip_off = draw_data.display_pos;
                        let clip_scale = draw_data.framebuffer_scale;

                        let left = draw_data.display_pos[0];
                        let right = draw_data.display_pos[0] + draw_data.display_size[0];
                        let top = draw_data.display_pos[1];
                        let bottom = draw_data.display_pos[1] + draw_data.display_size[1];
                        let matrix = [
                            [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                            [0.0, (2.0 / (top - bottom)), 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [
                                (right + left) / (left - right),
                                (top + bottom) / (bottom - top),
                                0.0,
                                1.0,
                            ],
                        ];

                        // Identify the current constant buffer offset before we write any new data into it
                        let dword_offset = constant_writer.dword_offset();

                        // Write the imgui matrix into the buffer
                        for row in &matrix {
                            for val in row {
                                constant_writer.write_all(&val.to_le_bytes()).unwrap();
                            }
                        }

                        for (idx, draw_list) in draw_data.draw_lists().enumerate() {
                            device.cmd_bind_vertex_buffers(
                                cmd_buffer,
                                0,
                                &[vtx_buffer_raw],
                                &[vtx_buffer_offsets[idx] as u64],
                            );

                            device.cmd_bind_index_buffer(
                                cmd_buffer,
                                idx_buffer_raw,
                                idx_buffer_offsets[idx] as u64,
                                vk::IndexType::UINT16,
                            );

                            for cmd in draw_list.commands() {
                                match cmd {
                                    DrawCmd::Elements {
                                        count,
                                        cmd_params:
                                            DrawCmdParams {
                                                clip_rect,
                                                texture_id,
                                                vtx_offset,
                                                idx_offset,
                                            },
                                    } => {
                                        let clip_rect = [
                                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                                        ];

                                        if clip_rect[0] < fb_width
                                            && clip_rect[1] < fb_height
                                            && clip_rect[2] >= 0.0
                                            && clip_rect[3] >= 0.0
                                        {
                                            let scissor_x =
                                                f32::max(0.0, clip_rect[0]).floor() as i32;
                                            let scissor_y =
                                                f32::max(0.0, clip_rect[1]).floor() as i32;
                                            let scissor_w =
                                                (clip_rect[2] - clip_rect[0]).abs().ceil() as u32;
                                            let scissor_h =
                                                (clip_rect[3] - clip_rect[1]).abs().ceil() as u32;

                                            device.cmd_set_scissor(
                                                cmd_buffer,
                                                0,
                                                &[vk::Rect2D::builder()
                                                    .offset(
                                                        vk::Offset2D::builder()
                                                            .x(scissor_x)
                                                            .y(scissor_y)
                                                            .build(),
                                                    )
                                                    .extent(
                                                        vk::Extent2D::builder()
                                                            .width(scissor_w)
                                                            .height(scissor_h)
                                                            .build(),
                                                    )
                                                    .build()],
                                            );

                                            // The texture slot index is stored inside the ImGui texture id
                                            let texture_index: u32 = texture_id.id() as u32;
                                            let push_constant_0 = ((texture_index & 0xff) << 24)
                                                | (dword_offset & 0x00ffffff);
                                            device.cmd_push_constants(
                                                cmd_buffer,
                                                renderer.pipeline_layout.raw(),
                                                vk::ShaderStageFlags::VERTEX
                                                    | vk::ShaderStageFlags::FRAGMENT
                                                    | vk::ShaderStageFlags::COMPUTE,
                                                0,
                                                &push_constant_0.to_le_bytes(),
                                            );

                                            device.cmd_draw_indexed(
                                                cmd_buffer,
                                                count as u32,
                                                1,
                                                idx_offset as u32,
                                                vtx_offset as i32,
                                                0,
                                            );
                                        }
                                    }
                                    DrawCmd::ResetRenderState => (), // NOTE: This doesn't seem necessary given how pipelines work?
                                    DrawCmd::RawCallback { callback, raw_cmd } => {
                                        callback(draw_list.raw(), raw_cmd)
                                    }
                                }
                            }
                        }
                    }

                    renderer.end_render();
                    drop(begin_render_enter);

                    renderer.end_frame(cmd_buffer);
                    drop(begin_frame_enter);
                }
                Event::LoopDestroyed => {
                    renderer.wait_for_idle();

                    imgui_vtx_buffers.clear();
                    imgui_idx_buffers.clear();
                }
                event => match event {
                    Event::DeviceEvent { event, .. } => match event {
                        DeviceEvent::Key(KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        }) => match (keycode, state) {
                            (VirtualKeyCode::Escape, ElementState::Released) => {
                                *control_flow = ControlFlow::Exit
                            }
                            _ => (),
                        },
                        _ => (),
                    },
                    _ => {}
                },
            }
        });
    }
}

#[derive(Debug, Clap)]
#[clap(version)]
struct SimOptions {
    /// Path to a RISC-V elf to execute
    elf_path: String,
}

// When this object is dropped, it flushes any logs that need flushing before closing the app
#[allow(dead_code)]
struct LogFlushGuards {
    chrome_guard: Option<tracing_chrome::FlushGuard>,
}

fn init_logging() -> LogFlushGuards {
    use tracing::event;
    use tracing_subscriber::{prelude::*, registry::Registry};

    let registry = Registry::default();

    // Line-oriented text output to stdout
    let registry = registry.with(tracing_subscriber::fmt::layer());

    // Output for `chrome://tracing`
    use tracing_chrome::ChromeLayerBuilder;

    // Place all traces in their own folder
    match std::fs::create_dir("./traces/") {
        Ok(()) => {}
        Err(err) => {
            // Don't let this stop us from running - we're probably just on read-only storage
            eprintln!(
                "Failed to create traces directory, no chrome://tracing traces: {:#?}",
                err
            );
        }
    }
    let trace_filename = format!(
        "./traces/trace-{}.json",
        std::time::SystemTime::UNIX_EPOCH
            .elapsed()
            .unwrap()
            .as_secs()
    );

    // Build the tracing-chrome layer
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(trace_filename).build();
    let registry = registry.with(chrome_layer);
    let chrome_guard = Some(guard);

    // Register our tracing subscriber
    tracing::subscriber::set_global_default(registry)
        .expect("Failed to install the tracing subscriber");

    let logging_test = tracing::trace_span!("Logging Test");
    let _logging_test_enter = logging_test.enter();

    event!(Level::ERROR, "1/5 logging checks");
    event!(Level::WARN, "2/5 logging checks");
    event!(Level::INFO, "3/5 logging checks");
    event!(Level::DEBUG, "4/5 logging checks");
    event!(Level::TRACE, "5/5 logging checks");

    LogFlushGuards { chrome_guard }
}

fn main() {
    let _logging_guard = init_logging();
    let opts = SimOptions::parse();
    show(&opts);
}
