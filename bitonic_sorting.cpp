#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h> // AVX2
#include <omp.h>
#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <malloc.h> // for _aligned_malloc on Windows
#include <climits>  // for INT_MAX
#include <pthread.h> // For pthreads

// 对齐分配（Windows兼容版）
int* aligned_alloc_int(size_t n) {
    return (int*)_aligned_malloc(n * sizeof(int), 32);
}

// 对齐释放
void aligned_free_int(int* ptr) {
    _aligned_free(ptr);
}

// 计算补齐2次幂长度
size_t roundup_pow2(size_t n) {
    if (n == 0) return 1; // Or handle as an error/special case
    size_t pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    return pow2;
}

// 填充补齐部分为 INT_MAX
void pad_array(int* arr, size_t n, size_t padded_n) {
    for (size_t i = n; i < padded_n; i++)
        arr[i] = INT_MAX;
}

// --- SIMD Merge (used by OpenMP, Pthread, and SIMD Only versions) ---
inline void simd_merge_step(int* arr, int low, int cnt, bool dir) {
    int half = cnt / 2;
    // Ensure half is a multiple of 8 for safe full vector processing,
    // or handle smaller blocks if cnt is small (e.g. 16, half is 8)
    for (int i = low; i < low + half; i += 8) {
        __m256i a = _mm256_load_si256((__m256i*)&arr[i]);
        __m256i b = _mm256_load_si256((__m256i*)&arr[i + half]);
        __m256i minv = _mm256_min_epi32(a, b);
        __m256i maxv = _mm256_max_epi32(a, b);
        if (dir) {
            _mm256_store_si256((__m256i*)&arr[i], minv);
            _mm256_store_si256((__m256i*)&arr[i + half], maxv);
        } else {
            _mm256_store_si256((__m256i*)&arr[i], maxv);
            _mm256_store_si256((__m256i*)&arr[i + half], minv);
        }
    }
}

const int SIMD_MERGE_SORT_THRESHOLD = 8; // Threshold for base case in SIMD merge

void bitonicMergeSIMD(int* arr, int low, int cnt, bool dir) {
    if (cnt <= 1) {
        return;
    }
    if (cnt <= SIMD_MERGE_SORT_THRESHOLD) {
        if (dir) {
            std::sort(arr + low, arr + low + cnt);
        } else {
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        }
        return;
    }
    // For cnt > SIMD_MERGE_SORT_THRESHOLD (e.g. > 8, so cnt is at least 16)
    simd_merge_step(arr, low, cnt, dir);
    int k = cnt / 2;
    bitonicMergeSIMD(arr, low, k, dir);
    bitonicMergeSIMD(arr, low + k, k, dir);
}

// --- 1. Serial Bitonic Sort (No SIMD in merge) ---
const int SERIAL_SORT_THRESHOLD = 32; // Threshold for std::sort in serial version

void bitonicMergeSerial(int* arr, int low, int cnt, bool dir) {
    if (cnt <= 1) return;

    if (cnt <= SERIAL_SORT_THRESHOLD) {
        if (dir) {
            std::sort(arr + low, arr + low + cnt);
        } else {
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        }
        return;
    }

    int k = cnt / 2;
    for (int i = low; i < low + k; ++i) {
        if (dir == (arr[i] > arr[i + k])) {
            std::swap(arr[i], arr[i + k]);
        }
    }
    bitonicMergeSerial(arr, low, k, dir);
    bitonicMergeSerial(arr, low + k, k, dir);
}

void bitonicSortSerialRecursive(int* arr, int low, int cnt, bool dir) {
    if (cnt <= 1) return;

    if (cnt <= SERIAL_SORT_THRESHOLD) {
        if (dir) {
            std::sort(arr + low, arr + low + cnt);
        } else {
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        }
        return;
    }

    int k = cnt / 2;
    bitonicSortSerialRecursive(arr, low, k, true);
    bitonicSortSerialRecursive(arr, low + k, k, false);
    bitonicMergeSerial(arr, low, cnt, dir);
}

void sort_wrapper_serial(int* arr_orig, size_t n) {
    size_t padded_n = roundup_pow2(n);
    if (padded_n == 0) return; 
    int* arr_aligned = aligned_alloc_int(padded_n);
    if (!arr_aligned) { std::cerr << "Memory allocation failed for serial sort." << std::endl; return; }
    
    memcpy(arr_aligned, arr_orig, n * sizeof(int));
    if (n < padded_n) pad_array(arr_aligned, n, padded_n);

    bitonicSortSerialRecursive(arr_aligned, 0, padded_n, true);

    memcpy(arr_orig, arr_aligned, n * sizeof(int));
    aligned_free_int(arr_aligned);
}

// --- 2. Pthread + SIMD Merge Bitonic Sort ---
const int PTHREAD_SORT_THRESHOLD = 32; 
const int PTHREAD_PARALLEL_DEPTH_LIMIT = 3; 

typedef struct {
    int* arr;
    int low;
    int cnt;
    bool dir; 
    int current_depth;
} BitonicPthreadArgs;

// Forward declaration
void bitonicSortPthread_SIMD_Recursive(int* arr, int low, int cnt, bool dir, int current_depth);

void* bitonicSortPthread_task(void* arg) {
    BitonicPthreadArgs* task_args = (BitonicPthreadArgs*)arg;
    bitonicSortPthread_SIMD_Recursive(task_args->arr, task_args->low, task_args->cnt, task_args->dir, task_args->current_depth);
    delete task_args; 
    return nullptr;
}

void bitonicSortPthread_SIMD_Recursive(int* arr, int low, int cnt, bool dir, int current_depth) {
    if (cnt <= PTHREAD_SORT_THRESHOLD) {
        if (dir)
            std::sort(arr + low, arr + low + cnt);
        else
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        return;
    }

    int k = cnt / 2;

    if (current_depth < PTHREAD_PARALLEL_DEPTH_LIMIT) {
        pthread_t tid1 = 0; 
        BitonicPthreadArgs* args1 = new BitonicPthreadArgs{arr, low, k, true, current_depth + 1};
        
        int rc = pthread_create(&tid1, nullptr, bitonicSortPthread_task, args1);
        
        if (rc != 0) { 
            delete args1; 
            bitonicSortPthread_SIMD_Recursive(arr, low, k, true, current_depth + 1); 
        }

        bitonicSortPthread_SIMD_Recursive(arr, low + k, k, false, current_depth + 1);

        if (rc == 0) { 
            pthread_join(tid1, nullptr);
        }
    } else {
        bitonicSortPthread_SIMD_Recursive(arr, low, k, true, current_depth + 1);
        bitonicSortPthread_SIMD_Recursive(arr, low + k, k, false, current_depth + 1);
    }
    bitonicMergeSIMD(arr, low, cnt, dir); // Use SIMD merge
}

void sort_wrapper_pthread_simd(int* arr_orig, size_t n) {
    size_t padded_n = roundup_pow2(n);
    if (padded_n == 0) return;
    int* arr_aligned = aligned_alloc_int(padded_n);
    if (!arr_aligned) { std::cerr << "Memory allocation failed for pthread+SIMD sort." << std::endl; return; }

    memcpy(arr_aligned, arr_orig, n * sizeof(int));
    if (n < padded_n) pad_array(arr_aligned, n, padded_n);

    bitonicSortPthread_SIMD_Recursive(arr_aligned, 0, padded_n, true, 0); 

    memcpy(arr_orig, arr_aligned, n * sizeof(int));
    aligned_free_int(arr_aligned);
}

// --- 3. OpenMP + SIMD Merge Bitonic Sort ---
const int OMP_SORT_THRESHOLD = 8;
const int OMP_PARALLEL_DEPTH_LIMIT = 3;

void bitonicSortSIMD_OMP_Recursive(int* arr, int low, int cnt, bool dir, int depth = 0) {
    if (cnt <= OMP_SORT_THRESHOLD) {
        if (dir)
            std::sort(arr + low, arr + low + cnt);
        else
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        return;
    }
    int k = cnt / 2;
    if (depth <= OMP_PARALLEL_DEPTH_LIMIT) {
        #pragma omp parallel sections
        {
            #pragma omp section
            bitonicSortSIMD_OMP_Recursive(arr, low, k, true, depth + 1);
            #pragma omp section
            bitonicSortSIMD_OMP_Recursive(arr, low + k, k, false, depth + 1);
        }
    } else {
        bitonicSortSIMD_OMP_Recursive(arr, low, k, true, depth + 1);
        bitonicSortSIMD_OMP_Recursive(arr, low + k, k, false, depth + 1);
    }
    bitonicMergeSIMD(arr, low, cnt, dir); // Use SIMD merge
}

void sort_wrapper_omp_simd(int* arr_orig, size_t n) {
    size_t padded_n = roundup_pow2(n);
    if (padded_n == 0) return;
    int* arr_aligned = aligned_alloc_int(padded_n);
    if (!arr_aligned) { std::cerr << "Memory allocation failed for OMP+SIMD sort." << std::endl; return; }

    memcpy(arr_aligned, arr_orig, n * sizeof(int));
    if (n < padded_n) pad_array(arr_aligned, n, padded_n);
    
    bitonicSortSIMD_OMP_Recursive(arr_aligned, 0, padded_n, true);
    
    memcpy(arr_orig, arr_aligned, n * sizeof(int));
    aligned_free_int(arr_aligned);
}

// --- 4. SIMD Only (Serial Control Flow) Bitonic Sort ---
const int SIMD_ONLY_SORT_THRESHOLD = 8; // Threshold for std::sort before SIMD merge

void bitonicSortSIMD_Only_Recursive(int* arr, int low, int cnt, bool dir) {
    if (cnt <= 1) return;

    if (cnt <= SIMD_ONLY_SORT_THRESHOLD) {
        if (dir) {
            std::sort(arr + low, arr + low + cnt);
        } else {
            std::sort(arr + low, arr + low + cnt, std::greater<int>());
        }
        return;
    }

    int k = cnt / 2;
    bitonicSortSIMD_Only_Recursive(arr, low, k, true);      // Sort first half ascending
    bitonicSortSIMD_Only_Recursive(arr, low + k, k, false); // Sort second half descending
    bitonicMergeSIMD(arr, low, cnt, dir);                   // Merge the two halves using SIMD
}

void sort_wrapper_simd_only(int* arr_orig, size_t n) {
    size_t padded_n = roundup_pow2(n);
    if (padded_n == 0) return;
    int* arr_aligned = aligned_alloc_int(padded_n);
    if (!arr_aligned) { std::cerr << "Memory allocation failed for SIMD Only sort." << std::endl; return; }

    memcpy(arr_aligned, arr_orig, n * sizeof(int));
    if (n < padded_n) pad_array(arr_aligned, n, padded_n);

    bitonicSortSIMD_Only_Recursive(arr_aligned, 0, padded_n, true); // Sort ascending

    memcpy(arr_orig, arr_aligned, n * sizeof(int));
    aligned_free_int(arr_aligned);
}


// --- Utility and Main ---
void generate_data(int* arr, size_t n) {
    std::mt19937 rng(std::random_device{}()); 
    std::uniform_int_distribution<int> dist(0, INT_MAX -1); 
    for (size_t i = 0; i < n; i++)
        arr[i] = dist(rng);
}

bool check_sorted(int* arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i-1] > arr[i]) {
            return false;
        }
    }
    return true;
}

template<typename Func>
void benchmark(Func sort_func, int* arr_orig_data, int* arr_copy, size_t n, const std::string& name) {
    if (n == 0) {
        std::cout << name << " time used: N/A (array size is 0)" << std::endl;
        return;
    }
    memcpy(arr_copy, arr_orig_data, n * sizeof(int)); 
    
    auto start = std::chrono::high_resolution_clock::now();
    sort_func(arr_copy, n); 
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_taken = std::chrono::duration<double>(end - start).count();
    bool correct = check_sorted(arr_copy, n); 
    
    std::cout << name << " time used: " << time_taken << " s, " 
              << (correct ? "right" : "error") << std::endl;
}

int main() {
    size_t N = 1500000; // Adjusted for potentially quicker testing
    // size_t N = 15000000; 

    if (N == 0) {
        std::cout << "N cannot be 0." << std::endl;
        return 1;
    }

    int* arr_original_data = aligned_alloc_int(N);
    int* arr_benchmark_copy = aligned_alloc_int(N);

    if (!arr_original_data || !arr_benchmark_copy) {
        std::cerr << "Failed to allocate memory for main arrays." << std::endl;
        if(arr_original_data) aligned_free_int(arr_original_data);
        if(arr_benchmark_copy) aligned_free_int(arr_benchmark_copy);
        return 1;
    }

    std::cout << "Generating " << N << " random numbers..." << std::endl;
    generate_data(arr_original_data, N);
    std::cout << "Data generation complete." << std::endl;
    std::cout << "Array Size (N): " << N << ", Padded Size: " << roundup_pow2(N) << std::endl;
    std::cout << "--- Benchmarks ---" << std::endl;

    benchmark(sort_wrapper_serial, arr_original_data, arr_benchmark_copy, N, "1. Serial (No SIMD Merge)");
    benchmark(sort_wrapper_simd_only, arr_original_data, arr_benchmark_copy, N, "4. SIMD Only (Serial Control, SIMD Merge)");
    benchmark(sort_wrapper_omp_simd, arr_original_data, arr_benchmark_copy, N, "3. OpenMP + SIMD Merge");
    benchmark(sort_wrapper_pthread_simd, arr_original_data, arr_benchmark_copy, N, "2. Pthread + SIMD Merge");
    
    aligned_free_int(arr_original_data);
    aligned_free_int(arr_benchmark_copy);
    
    std::cout << "------------------" << std::endl;
    std::cout << "Note: For Pthread version on Windows, compile with MinGW-w64 (g++) and link with -lpthread." << std::endl;
    std::cout << "Compile example (g++): g++ bitonic_sorting.cpp -o bitonic_sorting -O3 -mavx2 -fopenmp -lpthread" << std::endl;
    return 0;
}