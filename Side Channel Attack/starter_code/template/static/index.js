function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */
        this.isCollecting = true;
        this.status = "Collecting trace data...";
        this.statusIsError = false;

        try {
            // 1. Create a worker to run the sweep function
            let worker = new Worker("worker.js");

            // 2. Collect the trace data from the worker
            const result = await new Promise((resolve, reject) => {
                worker.onmessage = (e) => {
                    if (e.data.success) {
                        resolve(e.data);
                    } else {
                        reject(new Error(e.data.error || 'Worker failed'));
                    }
                };
                worker.onerror = (error) => reject(error);
                worker.postMessage("start");
            });

            // Terminate worker
            worker.terminate();

            this.status = "Sending trace data to backend...";

            // 3. Send the trace data to the backend for temporary storage and heatmap generation
            const response = await fetch('/collect_trace', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    traces: result.traces,
                    metadata: result.metadata
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const backendResult = await response.json();

            // 4. Fetch the heatmap from the backend and add it to the local collection
            if (backendResult.success) {
                this.heatmaps.push({
                    image: backendResult.heatmap_url,
                    metadata: result.metadata,
                    timestamp: new Date().toLocaleString()
                });
                
                // Also store in local collection
                this.traceData.push({
                    traces: result.traces,
                    metadata: result.metadata,
                    timestamp: new Date().toISOString()
                });

                this.status = `Trace collection complete! Collected ${result.traces.length} measurements.`;
            } else {
                throw new Error(backendResult.error || 'Backend processing failed');
            }

        } catch (error) {
            // 5. Handle errors and update the status
            console.error("Error collecting trace data:", error);
            this.status = `Error: ${error.message}`;
            this.statusIsError = true;
        } finally {
            this.isCollecting = false;
        }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
        this.status = "Preparing trace data for download...";
        this.statusIsError = false;

        try {
            // 1. Fetch the latest data from the backend API
            const response = await fetch('/api/get_traces', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // 2. Create a download file with the trace data in JSON format
            let downloadData;
            
            if (data.traces && data.traces.length > 0) {
                // Use backend data if available
                downloadData = {
                    traces: data.traces,
                    metadata: {
                        total_traces: data.traces.length,
                        collection_timestamp: new Date().toISOString(),
                        format: "array_of_arrays_for_ml"
                    }
                };
            } else if (this.traceData.length > 0) {
                // Fallback to local data
                downloadData = {
                    traces: this.traceData.map(trace => trace.traces),
                    metadata: {
                        total_traces: this.traceData.length,
                        collection_timestamp: new Date().toISOString(),
                        format: "array_of_arrays_for_ml",
                        trace_metadata: this.traceData.map(trace => trace.metadata)
                    }
                };
            } else {
                throw new Error('No trace data available for download');
            }

            // Create and trigger download
            const blob = new Blob([JSON.stringify(downloadData, null, 2)], { 
                type: 'application/json' 
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `traces_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.status = `Downloaded ${downloadData.traces.length} traces successfully!`;

        } catch (error) {
            // 3. Handle errors and update the status
            console.error("Error downloading traces:", error);
            this.status = `Error downloading traces: ${error.message}`;
            this.statusIsError = true;
        }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
      this.status = "Clearing all results...";
      this.statusIsError = false;

      try {
        // 1. Send a request to the backend API to clear all results
        const response = await fetch('/api/clear_results', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to clear results on server');
        }

        // 2. Clear local copies of trace data and heatmaps
        this.traceData = [];
        this.heatmaps = [];
        this.latencyResults = null;
        this.showingTraces = false;

        this.status = "All results cleared successfully!";

      } catch (error) {
        // 3. Handle errors and update the status
        console.error("Error clearing results:", error);
        this.status = `Error clearing results: ${error.message}`;
        this.statusIsError = true;
      }
    },
  };
}
