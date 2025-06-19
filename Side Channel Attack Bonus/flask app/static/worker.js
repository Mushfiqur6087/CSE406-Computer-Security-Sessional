/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
const LLCSIZE = 16777216;  // 16 MB from system config
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10; 

function sweep(P) {
    /*
     * Implement this function to run a sweep of the cache.
     * 1. Allocate a buffer of size LLCSIZE.
     * 2. Read each cache line (read the buffer in steps of LINESIZE).
     * 3. Count the number of times each cache line is read in a time period of P milliseconds.
     * 4. Store the count in an array of size K, where K = TIME / P.
     * 5. Return the array of counts.
     */
    
    try {
        // 1. Allocate a buffer of size LLCSIZE
        const buffer = new ArrayBuffer(LLCSIZE);
        const view = new Uint8Array(buffer);
        
        // Initialize buffer to ensure physical memory allocation
        for (let i = 0; i < LLCSIZE; i += LINESIZE) {
            view[i] = 1;
        }
        
        const K = Math.floor(TIME / P);  // Number of measurement periods
        const sweepCounts = new Array(K).fill(0);
        
        // 4. Store the count in an array of size K
        for (let period = 0; period < K; period++) {
            const startTime = performance.now();
            let sweepCount = 0;
            
            // Keep sweeping until P milliseconds have passed
            while ((performance.now() - startTime) < P) {
                // 2. Read each cache line (read the buffer in steps of LINESIZE)
                let sum = 0;
                for (let i = 0; i < LLCSIZE; i += LINESIZE) {
                    sum += view[i]; // Force memory access to each cache line
                }
                
                // 3. Count the number of times each cache line is read
                sweepCount++;
                
                // Prevent compiler optimization
                if (sum < 0) console.log("Impossible");
            }
            
            sweepCounts[period] = sweepCount;
        }
        
        // 5. Return the array of counts
        return sweepCounts;
        
    } catch (error) {
        console.error("Error in sweep function:", error);
        return null;
    }
}   

self.addEventListener('message', function(e) {
    /* Call the sweep function and return the result */
    if (e.data === 'start') {
        console.log('Starting sweep collection...');
        const result = sweep(P);
        
        if (result) {
            console.log(`Collected ${result.length} sweep measurements`);
            self.postMessage({
                success: true,
                traces: result,
                metadata: {
                    linesize: LINESIZE,
                    llcsize: LLCSIZE,
                    time: TIME,
                    period: P,
                    measurements: result.length
                }
            });
        } else {
            self.postMessage({
                success: false,
                error: 'Failed to collect sweep data'
            });
        }
    }
});