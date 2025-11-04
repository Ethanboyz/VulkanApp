#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <fstream>

static constexpr int WINDOW_WIDTH{800};
static constexpr int WINDOW_HEIGHT{600};
static constexpr int MAX_FRAMES_IN_FLIGHT{2};

class HelloTriangleApplication {
public:
    void run() {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

private:
    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;

    GLFWwindow* window_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physical_device_ = nullptr;
    vk::raii::Device device_ = nullptr;                             // Logical device
    vk::PhysicalDeviceFeatures device_features_{};

    vk::Queue graphics_queue_;                                      // Interface to the device's graphics command queue
    uint32_t graphics_queue_index_{};
    vk::Queue present_queue_;                                       // Interface to the device's present command queue
    uint32_t present_queue_index_{};

    vk::raii::SwapchainKHR swap_chain_ = nullptr;
    std::vector<vk::Image> swap_chain_images_;
    vk::Format swap_chain_format_ = vk::Format::eUndefined;
    vk::Extent2D swap_chain_extent_;
    std::vector<vk::raii::ImageView> swap_chain_image_views_;

    vk::raii::PipelineLayout pipeline_layout_ = nullptr;
    vk::raii::Pipeline graphics_pipeline_ = nullptr;

    vk::raii::CommandPool command_pool_ = nullptr;
    std::vector<vk::raii::CommandBuffer> command_buffers_;          // One command buffer per in-flight frame

    // Draw frame synchronization objects for queue submits and frame presents (one set per in-flight frame)
    std::vector<vk::raii::Semaphore> present_complete_semaphores_;  // For each in-flight frame, signaled when frame is presented to screen
    std::vector<vk::raii::Semaphore> render_complete_semaphores_;   // For each in-flight frame, signaled when frame is rendered, ready to be presented
    std::vector<vk::raii::Fence> in_flight_fences_;                 // For each in-flight frame, indicates when draw is done
    uint32_t current_frame_{};                                       // Current in-flight frame

    // Create vk instance, check for glfw extensions and validation layer support, if needed
    void create_instance() {
        constexpr vk::ApplicationInfo app_info {
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion13
        };

        // Get required glfw extensions, check if they are all supported by the instance
        uint32_t num_glfw_extensions{};
        auto glfw_required_extensions{glfwGetRequiredInstanceExtensions(&num_glfw_extensions)};
        auto extension_properties{context_.enumerateInstanceExtensionProperties()};
        for (uint32_t i = 0; i < num_glfw_extensions; i++) {
            const bool supported = std::ranges::any_of(
                extension_properties,
                [glfw_extension = glfw_required_extensions[i]](const auto& extension_property) {
                    return strcmp(extension_property.extensionName, glfw_extension) == 0;
                }
            );
            if (!supported) {
                throw std::runtime_error("Required GLFW extension not supported: " + std::string(glfw_required_extensions[i]));
            }
        }

        // Validation layers
        const std::vector required_validation_layers = {
            "VK_LAYER_KHRONOS_validation"
        };
        #ifdef NDEBUG
        constexpr bool enableValidationLayers{false};
        #else
        constexpr bool enable_validation_layers{true};
        #endif

        // Check if required validation layers are supported by the instance
        auto layer_properties = context_.enumerateInstanceLayerProperties();
        if (enable_validation_layers) {
            for (auto required_validation_layer : required_validation_layers) {
                const bool supported = std::ranges::any_of(
                    layer_properties,
                    [required_validation_layer](const auto& layer_property) {
                        return strcmp(layer_property.layerName, required_validation_layer) == 0;
                    }
                );
                if (!supported) {
                    throw std::runtime_error("Required validation layer not supported: " + std::string(required_validation_layer));
                }
            }
        }

        const vk::InstanceCreateInfo instance_create_info {
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(required_validation_layers.size()),
            .ppEnabledLayerNames = required_validation_layers.data(),
            .enabledExtensionCount = num_glfw_extensions,
            .ppEnabledExtensionNames = glfw_required_extensions
        };
        instance_ = vk::raii::Instance(context_, instance_create_info);
    }

    // Simply prioritize the first device that is a discrete GPU, otherwise pick whatever has Vulkan 1.3 support
    void pick_physical_device() {
        const auto devices = instance_.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("No GPUs with Vulkan support available!");
        }

        for (const auto& device : devices) {
            if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu && device.getProperties().apiVersion >= VK_API_VERSION_1_3) {
                physical_device_ = vk::raii::PhysicalDevice(device);
                return;
            }
        }
        for (const auto& device : devices) {
            if (device.getProperties().apiVersion >= VK_API_VERSION_1_3) {
                physical_device_ = vk::raii::PhysicalDevice(device);
                return;
            }
        }
        throw std::runtime_error("No GPUs with Vulkan >=1.3 support available!");
    }

    // Query and configure swap chain details to create a new swap chain
    void create_swap_chain() {
        const auto surface_capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_);
        vk::Extent2D swap_chain_extent;
        auto swap_chain_min_image_count = std::max(3u, surface_capabilities.minImageCount);

        // Make sure we didn't set swap_chain_min_image_count to be greater than the max (if specified) supported by the surface
        if (surface_capabilities.maxImageCount > 0 && swap_chain_min_image_count > surface_capabilities.maxImageCount) {
            swap_chain_min_image_count = surface_capabilities.maxImageCount;
        }

        // Choose a surface format, prefer SRGB color space (otherwise just use the first available format)
        const auto available_formats = physical_device_.getSurfaceFormatsKHR(surface_);
        auto chosen_format = available_formats[0];
        for (const auto& available_format : available_formats) {
            if (available_format.format == vk::Format::eB8G8R8A8Srgb && available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                chosen_format = available_format;
            }
        }

        // Choose a presentation mode, prefer mailbox but use FIFO if not available
        const auto available_present_modes = physical_device_.getSurfacePresentModesKHR(surface_);
        auto chosen_present_mode = vk::PresentModeKHR::eFifo;
        for (const auto& available_present_mode : available_present_modes) {
            if (available_present_mode == vk::PresentModeKHR::eMailbox) {
                chosen_present_mode = available_present_mode;
            }
        }

        // Configure the swap chain image resolution, matching it to the glfw framebuffer size
        const bool image_size_already_specified = surface_capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max();
        if (image_size_already_specified) {
            swap_chain_extent = surface_capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window_, &width, &height);       // Window size in pixel coordinates
            swap_chain_extent = vk::Extent2D{
                std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height)
            };
        }

        // Put together everything into the new swap chain
        vk::SwapchainCreateInfoKHR swapchain_create_info {
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = surface_,
            .minImageCount = swap_chain_min_image_count,
            .imageFormat = chosen_format.format,
            .imageColorSpace = chosen_format.colorSpace,
            .imageExtent = swap_chain_extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = surface_capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = chosen_present_mode,
            .clipped = true,
            .oldSwapchain = nullptr
        };
        // Use concurrent sharing mode if graphics and present queue are separate (no ownership handling required but slower)
        const uint32_t queue_family_indices[] = {graphics_queue_index_, present_queue_index_};
        if (graphics_queue_index_ != present_queue_index_) {
            swapchain_create_info.imageSharingMode = vk::SharingMode::eConcurrent;
            swapchain_create_info.queueFamilyIndexCount = 2;
            swapchain_create_info.pQueueFamilyIndices = queue_family_indices;
        }

        swap_chain_ = vk::raii::SwapchainKHR(device_, swapchain_create_info);
        swap_chain_images_ = swap_chain_.getImages();
        swap_chain_format_ = chosen_format.format;
        swap_chain_extent_ = swap_chain_extent;
    }

    // Create a new logical device assuming a physical device was picked, enabling device-layer extensions and setting up its graphics command queue
    void create_logical_device() {
        // Search for a graphics and a present capable queue, prefer one queue that supports both capabilities
        const auto queue_families = physical_device_.getQueueFamilyProperties();
        uint32_t graphics_queue_index = queue_families.size();
        uint32_t present_queue_index = queue_families.size();
        for (int index = 0; index < queue_families.size(); index++) {
            const bool supports_graphics = static_cast<bool>(queue_families[index].queueFlags & vk::QueueFlagBits::eGraphics);
            const bool supports_present = physical_device_.getSurfaceSupportKHR(index, surface_);
            if (supports_graphics && supports_present) {
                graphics_queue_index = index;
                present_queue_index = index;
                break;
            }
            if (supports_graphics) {
                graphics_queue_index = index;
            } else if (supports_present) {
                present_queue_index = index;
            }
        }
        if (graphics_queue_index == queue_families.size() || present_queue_index == queue_families.size()) {
            throw std::runtime_error("Selected GPU does not support graphics or present capabilities!");
        }
        graphics_queue_index_ = graphics_queue_index;
        present_queue_index_ = present_queue_index;

        float queue_priority = 1.f;
        vk::DeviceQueueCreateInfo device_queue_create_info {
            .queueFamilyIndex = graphics_queue_index,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority
        };

        // Enable 1.0+ device features w/ a structure chain (feature structures w/ initializers), check if they are supported
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        > feature_chain = {
            {},
            {.shaderDrawParameters = true},
            {.synchronization2 = true, .dynamicRendering = true},
            {.extendedDynamicState = true}
        };
        auto supported_features = physical_device_.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        if (!supported_features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters
            || !supported_features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2
            || !supported_features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering
            || !supported_features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState) {
            throw std::runtime_error("Selected GPU does not support all required extensions!");
        }

        // Enable device extensions and check if they are supported by the physical device
        std::vector device_extensions = {
            vk::KHRSwapchainExtensionName,
            vk::KHRSpirv14ExtensionName,
            vk::KHRSynchronization2ExtensionName,
            vk::KHRCreateRenderpass2ExtensionName
        };
        const auto device_extension_properties = physical_device_.enumerateDeviceExtensionProperties();
        for (const auto& device_extension : device_extensions) {
            bool supported = std::ranges::any_of(
                device_extension_properties,
                [device_extension](auto extension) {
                    return std::strcmp(extension.extensionName, device_extension) == 0;
                }
            );
            if (!supported) {
                throw std::runtime_error("Selected GPU does not support device extension: " + std::string(device_extension));
            }
        }

        // Create the logical device
        const vk::DeviceCreateInfo device_create_info {
            .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),     // Points to full structure chain of features
            .queueCreateInfoCount = device_queue_create_info.queueCount,
            .pQueueCreateInfos = &device_queue_create_info,
            .enabledExtensionCount = static_cast<uint32_t>(device_extensions.size()),
            .ppEnabledExtensionNames = device_extensions.data()
        };
        device_ = vk::raii::Device(physical_device_, device_create_info);
        graphics_queue_ = vk::raii::Queue(device_, graphics_queue_index, 0);  // Assuming only one queue from this family will be used
        present_queue_ = vk::raii::Queue(device_, present_queue_index, 0);
    }

    // Create new window surface
    void create_surface() {
        VkSurfaceKHR surface;
        if (const VkResult result = glfwCreateWindowSurface(*instance_, window_, nullptr, &surface); result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface: ERROR CODE " + std::to_string(result));
        }
        surface_ = vk::raii::SurfaceKHR(instance_, surface);
    }

    // Creates an image view for every swap chain image
    void create_image_views() {
        swap_chain_image_views_.clear();
        vk::ImageViewCreateInfo image_view_create_info {
            .viewType = vk::ImageViewType::e2D,
            .format = swap_chain_format_,
            .subresourceRange = {
                vk::ImageAspectFlagBits::eColor,
                0,
                1,
                0,
                1
            }
        };

        for (const auto& image : swap_chain_images_) {
            image_view_create_info.image = image;
            swap_chain_image_views_.emplace_back(device_, image_view_create_info);
        }
    }

    // Read file into a vector of bytes
    static std::vector<char> read_file(const std::string& filename) {
        std::ifstream file{filename, std::ios::ate | std::ios::binary};    // Open in binary mode, seek to end of file
        if (!file) {
            throw std::runtime_error("Failed to open file " + filename);
        }
        const auto filesize = file.tellg();
        std::vector<char> buffer(filesize);

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }

    [[nodiscard]] vk::raii::ShaderModule create_shader_module(const std::vector<char>& spv_code) const {
        const vk::ShaderModuleCreateInfo shader_module_create_info {
            .codeSize = spv_code.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(spv_code.data()),
        };
        return {device_, shader_module_create_info};
    }

    // Initialize the graphics rasterization pipeline
    void create_graphics_pipeline() {
        vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        vk::PipelineVertexInputStateCreateInfo vertex_input_create_info;

        // Load shader programs
        const auto shaders = read_file("shaders/slang.spv");
        const auto shader_module = create_shader_module(shaders);
        const vk::PipelineShaderStageCreateInfo vertex_shader_stage_create_info {
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shader_module,
            .pName = "vertex_main"
        };
        const vk::PipelineShaderStageCreateInfo fragment_shader_stage_create_info {
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shader_module,
            .pName = "fragment_main"
        };
        const vk::PipelineShaderStageCreateInfo shader_stages[] = {vertex_shader_stage_create_info, fragment_shader_stage_create_info};

        // Mark viewport and scissor as dynamic for flexibility
        const std::vector dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        const vk::PipelineDynamicStateCreateInfo dynamic_state = {
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data()
        };
        constexpr vk::PipelineViewportStateCreateInfo viewport_state_create_info {
            .viewportCount = 1,
            .scissorCount = 1
        };

        // Set up the rasterizer
        constexpr vk::PipelineRasterizationStateCreateInfo rasterization_create_info = {
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = vk::False,
            .depthBiasSlopeFactor = 1.f,
            .lineWidth = 1.f
        };

        // Disable multisampling for now
        constexpr vk::PipelineMultisampleStateCreateInfo multisample_create_info {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        // Color blending
        vk::PipelineColorBlendAttachmentState color_blend_attachment;
        color_blend_attachment.colorWriteMask =
            vk::ColorComponentFlagBits::eR
            | vk::ColorComponentFlagBits::eG
         | vk::ColorComponentFlagBits::eB
         | vk::ColorComponentFlagBits::eA;
        color_blend_attachment.blendEnable = vk::False;
        const vk::PipelineColorBlendStateCreateInfo color_blend_create_info {
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment
        };

        // Pipeline layout creation and render pass
        constexpr vk::PipelineLayoutCreateInfo pipeline_layout_create_info = {
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0
        };
        pipeline_layout_ = vk::raii::PipelineLayout{device_, pipeline_layout_create_info};

        vk::PipelineRenderingCreateInfo pipeline_rendering_create_info {    // Allows dynamic rendering
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &swap_chain_format_
        };
        vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info {
            .pNext = &pipeline_rendering_create_info,
            .stageCount = 2,
            .pStages = shader_stages,
            .pVertexInputState = &vertex_input_create_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state_create_info,
            .pRasterizationState = &rasterization_create_info,
            .pMultisampleState = &multisample_create_info,
            .pColorBlendState = &color_blend_create_info,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout_,
            .renderPass = nullptr,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1
        };
        graphics_pipeline_ = vk::raii::Pipeline{device_, nullptr, graphics_pipeline_create_info};
    }

    // Create the command pool for the graphics queue family
    void create_command_pool() {
        vk::CommandPoolCreateInfo command_pool_create_info {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = graphics_queue_index_
        };
        command_pool_ = {device_, command_pool_create_info};
    }

    // Allocates one command buffer per intended in-flight frame from the command pool
    void create_command_buffers() {
        command_buffers_.clear();
        const vk::CommandBufferAllocateInfo command_buffer_allocate_info {
            .commandPool = command_pool_,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };
        command_buffers_ = vk::raii::CommandBuffers(device_, command_buffer_allocate_info);
    }

    // Transition the image layout from one layout to another
    void transition_image_layout(
        const uint32_t swap_chain_image_index,
        const vk::ImageLayout old_layout,
        const vk::ImageLayout new_layout,
        const vk::AccessFlags2 src_access_mask,
        const vk::AccessFlags2 dst_access_mask,
        const vk::PipelineStageFlags2 src_stage_mask,
        const vk::PipelineStageFlags2 dst_stage_mask
    ) const {
        const vk::ImageMemoryBarrier2 image_memory_barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swap_chain_images_[swap_chain_image_index],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        const vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_memory_barrier
        };
        command_buffers_[current_frame_].pipelineBarrier2(dependency_info);
    }

    // Record commands to the allocated command pool buffer through the graphics pipeline
    void record_command_buffer(const uint32_t swap_chain_image_index) const {
        command_buffers_[current_frame_].begin({});      // Start recording the allocated command buffer

        // Transition swap chain image to color attachment layout
        transition_image_layout(
            swap_chain_image_index,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        // Set up color attachment, we want to wipe the specified swap chain image black before rendering and storing it
        constexpr vk::ClearValue clear_color = vk::ClearColorValue(0.f, 0.f, 0.f, 1.f);
        const vk::RenderingAttachmentInfo rendering_attachment_info {
            .imageView = swap_chain_image_views_[swap_chain_image_index],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clear_color
        };

        // Set up rendering
        const vk::RenderingInfo rendering_info {
            .renderArea = {
                .offset = {0, 0},
                .extent = swap_chain_extent_
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &rendering_attachment_info
        };
        command_buffers_[current_frame_].beginRendering(rendering_info);
        command_buffers_[current_frame_].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline_);

        const auto viewport = vk::Viewport(
            0.f,
            0.f,
            static_cast<float>(swap_chain_extent_.width),
            static_cast<float>(swap_chain_extent_.height),
            0.f,
            1.f
        );
        const auto scissors = vk::Rect2D{
            .offset = vk::Offset2D(0, 0),
            .extent = swap_chain_extent_
        };
        command_buffers_[current_frame_].setViewport(0, viewport);
        command_buffers_[current_frame_].setScissor(0, scissors);

        command_buffers_[current_frame_].draw(3, 1, 0, 0);

        // Done rendering
        command_buffers_[current_frame_].endRendering();

        // Transition swap chain image to present src layout
        transition_image_layout(
            swap_chain_image_index,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        command_buffers_[current_frame_].end();
    }

    // Initialize the necessary semaphores and fences for ordered frame draws
    void create_sync_objects() {
        present_complete_semaphores_.clear();
        render_complete_semaphores_.clear();
        in_flight_fences_.clear();

        constexpr vk::FenceCreateInfo fence_create_info = {
            .flags = vk::FenceCreateFlagBits::eSignaled
        };
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            present_complete_semaphores_.emplace_back(device_, vk::SemaphoreCreateInfo());
            render_complete_semaphores_.emplace_back(device_, vk::SemaphoreCreateInfo());
            in_flight_fences_.emplace_back(device_, fence_create_info);
        }
    }

    // Complete the render of a single frame
    void draw_frame() {
        // Wait until the GPU is done rendering the previous frame
        while (device_.waitForFences(*in_flight_fences_[current_frame_], vk::True, UINT64_MAX) == vk::Result::eTimeout) {}

        // Acquire image from swap chain, write command
        auto [result, swap_chain_image_index] = swap_chain_.acquireNextImage(UINT64_MAX, *present_complete_semaphores_[current_frame_], nullptr);
        device_.resetFences(*in_flight_fences_[current_frame_]);

        command_buffers_[current_frame_].reset();
        record_command_buffer(swap_chain_image_index);

        // Submit command buffer to the graphics queue
        constexpr vk::PipelineStageFlags pipeline_stage_flags{vk::PipelineStageFlagBits::eColorAttachmentOutput};
        const vk::SubmitInfo submit_info = {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*present_complete_semaphores_[current_frame_],
            .pWaitDstStageMask = &pipeline_stage_flags,
            .commandBufferCount = 1,
            .pCommandBuffers = &*command_buffers_[current_frame_],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*render_complete_semaphores_[current_frame_]
        };
        graphics_queue_.submit(submit_info, *in_flight_fences_[current_frame_]);

        // Tell the GPU to present the result, but only after the rendering is complete
        const vk::PresentInfoKHR present_info_khr = {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*render_complete_semaphores_[current_frame_],
            .swapchainCount = 1,
            .pSwapchains = &*swap_chain_,
            .pImageIndices = &swap_chain_image_index
        };
        result = present_queue_.presentKHR(present_info_khr);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("vk::Queue::presentKHR returned non-success: " + vk::to_string(result));
        }
        current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Initialize the glfw window
    void init_window() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "HelloTriangleApp", nullptr, nullptr);
    }

    void init_vulkan() {
        create_instance();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swap_chain();
        create_image_views();
        create_graphics_pipeline();
        create_command_pool();
        create_command_buffers();
        create_sync_objects();
    }

    void main_loop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            draw_frame();
        }
        device_.waitIdle();
    }

    void cleanup() const {
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}