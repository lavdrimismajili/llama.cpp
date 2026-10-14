#include <sycl/sycl.hpp>
int main() {
    sycl::queue q(sycl::gpu_selector_v);
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
    q.submit([&](sycl::handler& h) {
        h.parallel_for(64, [=](sycl::id<1>) {
            std::cout << "Sub-group size: " << sycl::this_work_group.get_local_range(0) << "\n";
        });
    }).wait();
    // Or more directly:
    std::cout << "Max sub-group: " << dev.get_info<sycl::info::device::max_sub_group_size>() << "\n";
    return 0;
}