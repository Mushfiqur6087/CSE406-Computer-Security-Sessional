/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */
  
  try {
    // 1. Allocate a buffer of size n * LINESIZE
    const bufferSize = n * LINESIZE;
    const buffer = new ArrayBuffer(bufferSize);
    const view = new Uint8Array(buffer);
    
    // Initialize buffer with some data to ensure it's allocated
    for (let i = 0; i < bufferSize; i += LINESIZE) {
      view[i] = 1;
    }
    
    const timings = [];
    
    // 2. Read the entire buffer 10 times and measure timing
    for (let repeat = 0; repeat < 10; repeat++) {
      const startTime = performance.now();
      
      // Read each cache line (at intervals of LINESIZE)
      let sum = 0;
      for (let i = 0; i < bufferSize; i += LINESIZE) {
        sum += view[i]; // Force memory access
      }
      
      const endTime = performance.now();
      timings.push(endTime - startTime);
      
      // Prevent optimization by using the sum
      if (sum < 0) console.log("Impossible");
    }
    
    // 4. Return the median of the time taken
    timings.sort((a, b) => a - b);
    const median = timings[Math.floor(timings.length / 2)];
    return median;
    
  } catch (error) {
    // Return null if allocation fails (e.g., out of memory)
    return null;
  }
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    
    // Test with powers of 10: 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000
    for (let n = 1; n <= 10000000; n *= 10) {
      const timing = readNlines(n);
      
      // If the function fails (returns null), break the loop
      if (timing === null) {
        console.log(`Failed to allocate memory for n = ${n}, stopping measurements`);
        break;
      }
      
      results[n] = timing;
      
      // Also test some intermediate values for more granular data
      if (n < 10000000) {
        // Test n * 5 as well (5, 50, 500, etc.)
        const intermediateN = n * 5;
        const intermediateTiming = readNlines(intermediateN);
        if (intermediateTiming !== null) {
          results[intermediateN] = intermediateTiming;
        }
      }
    }

    self.postMessage(results);
  }
});

/*
 * MEMORY INITIALIZATION EXPLANATION:
 * 
 * Why we initialize the buffer with view[i] = 1 before timing measurements:
 * 
 * 1. VIRTUAL vs PHYSICAL MEMORY ALLOCATION:
 *    - new ArrayBuffer(bufferSize) only reserves virtual address space
 *    - Physical RAM pages aren't allocated until first access (lazy allocation)
 *    - This is called "demand paging" or "copy-on-write"
 * 
 * 2. PAGE FAULT OVERHEAD:
 *    Without initialization, first read of each 4KB page incurs:
 *    - Page fault cost (OS maps virtual → physical memory)
 *    - Zero-fill cost (OS provides clean memory page)
 *    - Cache miss cost (load 64-byte cache line into L1/L2/L3)
 *    - DRAM fetch cost (actual memory read)
 * 
 * 3. TIMING MEASUREMENT DISTORTION:
 *    Without initialization:
 *    - Pass #1: Measures "page fault + cache miss + DRAM fetch"
 *    - Pass #2-10: Measures only "cache hit/miss + DRAM fetch"
 *    - Result: Inconsistent timing data, skewed median values
 * 
 *    With initialization:
 *    - All passes measure purely "cache hierarchy vs RAM" behavior
 *    - Consistent, reliable timing measurements
 *    - True cache performance characteristics
 * 
 * 4. ATTACK ACCURACY:
 *    - Pure cache timing = better website fingerprinting
 *    - Mixed page fault timing = unreliable attack results
 *    - Initialization ensures we measure only cache behavior
 */