#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

static constexpr int WINDOW_WIDTH{800};
static constexpr int WINDOW_HEIGHT{600};

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
    vk::raii::Device device_ = nullptr;         // Logical device
    vk::PhysicalDeviceFeatures device_features_{};
    vk::Queue graphics_queue_;                  // Interface to the device's graphics command queue
    vk::Queue present_queue_;                   // Interface to the device's present command queue

    // Create vk instance, check for glfw extensions and validation layer support, if needed
    void createInstance() {
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

        vk::DeviceQueueCreateInfo device_queue_create_info {
            .queueFamilyIndex = graphics_queue_index,
            .queueCount = 1,
            .pQueuePriorities = new float(0.f)
        };

        // Enable 1.0+ device features w/ a structure chain (three feature structures w/ initializers)
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> feature_chain = {
            {},
            {.dynamicRendering = true},
            {.extendedDynamicState = true}
        };

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
        const VkResult result = glfwCreateWindowSurface(*instance_, window_, nullptr, &surface);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface: ERROR CODE " + std::to_string(result));
        }
        surface_ = vk::raii::SurfaceKHR(instance_, surface);
    }

    void init_window() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "HelloTriangleApp", nullptr, nullptr);
    }

    void init_vulkan() {
        createInstance();
        create_surface();
        pick_physical_device();
        create_logical_device();
    }

    void main_loop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
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