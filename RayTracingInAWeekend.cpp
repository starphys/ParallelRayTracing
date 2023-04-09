
#include "rtweekend.h"

#include "box.h"
#include "bvh.h"
#include "camera.h"
#include "color.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "scenes.h"
#include "sphere.h"
#include "texture.h"

#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>


color ray_color(const ray& r, const color& background, const hittable& world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0, 0, 0);

    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec))
        return background;

    ray scattered;
    color attenuation;
    color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
        return emitted;

    return emitted + attenuation * ray_color(scattered, background, world, depth - 1);
}

struct Job
{
    int startIndex{0};
    int endIndex{0};
    int height;
    int width;
    int samples;
    int depth;
    std::vector<int> indices;
    std::vector<color> colors;
};

// Complete a single job and store result
void RenderJob(Job job, std::vector<Job>& results,
    camera cam, hittable_list world, color background,
    std::mutex& mutex) {
    for (int j = job.startIndex; j < job.endIndex; ++j) {

        for (int i = 0; i < job.width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < job.samples; ++s) {
                auto u = (i + random_double()) / (job.width - 1);
                auto v = (j + random_double()) / (job.height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, background, world, job.depth);
            }

            job.indices.push_back(j*job.width+i);
            job.colors.push_back(pixel_color);
        }
    }

    // Job is done, record it in results
    {
        std::lock_guard<std::mutex> lock(mutex);
        results.push_back(job);
    }
}

// Keep threads working until all jobs are completed
void ScheduleJobs(std::vector<Job>& results, std::queue<Job>& todo, 
    camera cam, hittable_list world, color background,
    std::mutex& mutex, std::condition_variable& finished) {

    // Set when all jobs have been dispatched
    std::atomic<bool> done{false};

    while (!done) {
        Job newJob;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!todo.empty()) {
                newJob = todo.front();
                todo.pop();
            }
        }

        if (newJob.startIndex < newJob.endIndex) {
            RenderJob(newJob, results, cam, world, background, mutex);
        }
        // Otherwise we've finished all of our jobs
        else {
            done = true;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        finished.notify_one();
    }


}


int main() {
    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    int samples_per_pixel = 10;
    int max_depth = 50;

    // World

    hittable_list world;

    point3 lookfrom;
    point3 lookat;
    auto vfov = 40.0;
    auto aperture = 0.0;
    color background(0, 0, 0);

    switch (1) {
    case 1:
        world = random_scene();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 30.0;
        aperture = 0.1;
        break;

    case 2:
        world = two_spheres();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        break;

    case 3:
        world = two_perlin_spheres();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        break;

    case 4:
        world = earth();
        background = color(0.70, 0.80, 1.00);
        lookfrom = point3(0, 0, 12);
        lookat = point3(0, 0, 0);
        vfov = 20.0;
        break;

    case 5:
        world = simple_light();
        samples_per_pixel = 400;
        lookfrom = point3(26, 3, 6);
        lookat = point3(0, 2, 0);
        vfov = 20.0;
        break;

    default:
    case 6:
        world = cornell_box();
        aspect_ratio = 1.0;
        image_width = 600;
        samples_per_pixel = 200;
        lookfrom = point3(278, 278, -800);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;

    case 7:
        world = cornell_smoke();
        aspect_ratio = 1.0;
        image_width = 600;
        samples_per_pixel = 200;
        lookfrom = point3(278, 278, -800);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;

    case 8:
        world = final_scene();
        aspect_ratio = 1.0;
        image_width = 600;
        samples_per_pixel = 500;
        lookfrom = point3(478, 278, -600);
        lookat = point3(278, 278, 0);
        vfov = 40.0;
        break;
    }
    // Camera

    const vec3 vup(0, 1, 0);
    const auto dist_to_focus = 10.0;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int pixels = image_height * image_width;

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    clock_t start, stop;
    start = clock();

    // Render - split the work up into jobs, schedule the jobs to available threads.
    color* image = new color[pixels];

    std::mutex mutex;
    std::condition_variable finished;
    std::vector<Job> results;
    std::queue<Job> todo;
    std::vector<std::thread> threads;

    unsigned int threadCount = std::thread::hardware_concurrency();
    
    //TODO: hardcoding vs algorithmic values
    int rowsPerJob = 20;
    int nJobs = image_height / rowsPerJob;
    int remJobs = image_height % rowsPerJob;

    // Create jobs
    for (int i = 0; i < nJobs; ++i)
    {
        Job job;
        job.startIndex = i * rowsPerJob;
        job.endIndex = job.startIndex + rowsPerJob;
        if (i == nJobs - 1)
        {
            // Last job is slightly larger
            job.endIndex += remJobs;
        }
        job.height = image_height;
        job.width = image_width;
        job.depth = max_depth;
        job.samples = samples_per_pixel;

        todo.push(job);
    }

    // Spin up threads, make main/parent the last thread
    for (int i = 0; i < threadCount - 1; ++i)
    {
        std::thread t([&]() {
            ScheduleJobs(results, todo, cam, world, background, mutex, finished);
            });
        threads.push_back(std::move(t));
    }

    ScheduleJobs(results, todo, cam, world, background, mutex, finished);

    // Wait for all jobs to be done in all threads
    {
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock, [&results, &nJobs] {
            return results.size() == nJobs;
        });
    }

    for (std::thread& t : threads)
    {
        t.join();
    }

    // Finalize the image

    for (Job job : results)
    {
        int colorIndex = 0;

        for (int i = 0; i < job.colors.size(); ++i)
        {
            int image_index = job.indices[i];
            image[image_index] = job.colors[i];
        }
    }
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int i = pixels - 1; i >= 0; --i)
    {
        write_color(std::cout, image[i], samples_per_pixel);
    }

    std::cerr << "\nDone.\n";

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    delete[] image;
    return 0;
}