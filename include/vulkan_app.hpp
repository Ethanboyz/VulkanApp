#ifndef VULKAN_APP_H
#define VULKAN_APP_H

#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/fwd.hpp>
#include <vector>

static constexpr int WINDOW_WIDTH{800};
static constexpr int WINDOW_HEIGHT{600};
static constexpr int MAX_FRAMES_IN_FLIGHT{2};

struct Vertex {
    glm::vec2 position;
    glm::vec3 color;

    static vk::VertexInputBindingDescription get_binding_description() {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }
    static std::array<vk::VertexInputAttributeDescription, 2> get_attribute_descriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
        };
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

const std::vector<Vertex> vertices {
    {{-0.5, -0.5}, {1.0, 1.0, 1.0}},    // Upper left
    {{0.5, -0.5}, {1.0, 1.0, 1.0}},     // Upper right
    {{0.5, 0.5}, {0.0, 1.0, 0.0}},      // Lower right
    {{-0.5, 0.5}, {0.0, 0.0, 1.0}},     // Lower left
};

const std::vector<uint16_t> indices {
    0, 1, 2, 0, 2, 3
};

class VulkanApp {
public:
    void run();

private:
    // High-level workflow
    void init_window();
    void init_vulkan();
    void main_loop();
    void cleanup();

    // Initialization
    void create_instance();
    void create_surface();
    void pick_physical_device();
    void create_logical_device();
    void create_swap_chain();
    void create_image_views();
    void create_descriptor_set_layout();
    void create_graphics_pipeline();
    void create_command_pool();
    void create_vertex_buffer();
    void create_index_buffer();
    void create_command_buffers();
    void create_sync_objects();

    // Main loop
    void draw_frame();

    // Cleanup
    void cleanup_swap_chain();

    // Helpers
    void copy_buffer(const vk::raii::Buffer& src_buffer, vk::raii::Buffer& dst_buffer, vk::DeviceSize src_buffer_size); // Used to create and populate vertex, index buffers
    void transition_image_layout(   // Used in record_command_buffers(), transitions swap chain images so they can be used as render targets and then presented
        const uint32_t swap_chain_image_index,
        const vk::ImageLayout old_layout,
        const vk::ImageLayout new_layout,
        const vk::AccessFlags2 src_access_mask,
        const vk::AccessFlags2 dst_access_mask,
        const vk::PipelineStageFlags2 src_stage_mask,
        const vk::PipelineStageFlags2 dst_stage_mask
    ) const;
    void record_command_buffer(const uint32_t swap_chain_image_index) const;    // Records draw-related commands to command buffer(s)
    void recreate_swap_chain(); // Recreates the swap chain, useful for events where the current swap chain becomes invalid, like window resizing
    static void framebuffer_resize_callback(GLFWwindow* window, [[maybe_unused]] int window_width, [[maybe_unused]] int window_height);    // Called whenever the window is resized

    // Members
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

    // Graphics pipeline
    vk::raii::PipelineLayout pipeline_layout_ = nullptr;
    vk::raii::Pipeline graphics_pipeline_ = nullptr;

    vk::raii::Buffer vertex_buffer_ = nullptr;
    vk::raii::DeviceMemory vertex_buffer_memory_ = nullptr;

    vk::raii::Buffer index_buffer_ = nullptr;
    vk::raii::DeviceMemory index_buffer_memory_ = nullptr;

    vk::raii::CommandPool command_pool_ = nullptr;
    std::vector<vk::raii::CommandBuffer> command_buffers_;          // One command buffer per in-flight frame

    // Draw frame synchronization objects for queue submits and frame presents (one set per in-flight frame)
    std::vector<vk::raii::Semaphore> present_complete_semaphores_;  // For each in-flight frame, signaled when frame is presented to screen
    std::vector<vk::raii::Semaphore> render_complete_semaphores_;   // For each in-flight frame, signaled when frame is rendered, ready to be presented
    std::vector<vk::raii::Fence> in_flight_fences_;                 // For each in-flight frame, indicates when draw is done
    uint32_t current_frame_{};                                      // Current in-flight frame
    bool framebuffer_resized = false;                               // True if the resize has occurred
};

#endif