from flask import Flask, send_from_directory, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import os
import json
from datetime import datetime

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return the heatmap image and optionally other statistics to the frontend
    """
    try:
        # 1. Receive trace data from the frontend as JSON
        data = request.get_json()
        if not data or 'traces' not in data:
            return jsonify({'success': False, 'error': 'No trace data provided'}), 400
        
        traces = data['traces']
        metadata = data.get('metadata', {})
        
        # Validate trace data
        if not traces or not isinstance(traces, list):
            return jsonify({'success': False, 'error': 'Invalid trace data format'}), 400
        
        # 2. Generate a heatmap using matplotlib
        plt.figure(figsize=(12, 6))
        
        # Create a 2D representation of the trace data for visualization
        # Reshape the 1D trace array into a 2D matrix for better visualization
        trace_array = np.array(traces)
        
        # Create a matrix representation - we'll use multiple rows to show patterns
        rows = min(20, len(trace_array) // 10) or 1  # At least 1 row, max 20 rows
        cols = len(trace_array) // rows
        
        if cols > 0:
            # Reshape data to create a 2D heatmap
            reshaped_data = trace_array[:rows * cols].reshape(rows, cols)
        else:
            # Fallback: create a single row
            reshaped_data = trace_array.reshape(1, -1)
        
        # Create the heatmap
        plt.imshow(reshaped_data, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Sweep Count')
        plt.title(f'Cache Sweep Trace Heatmap\n{len(traces)} measurements, Period: {metadata.get("period", "N/A")}ms')
        plt.xlabel('Time Window')
        plt.ylabel('Trace Segment')
        
        # Add some statistics as text
        stats_text = f'Min: {np.min(trace_array):.1f}, Max: {np.max(trace_array):.1f}, Mean: {np.mean(trace_array):.1f}'
        plt.figtext(0.02, 0.02, stats_text, fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 3. Store the heatmap and trace data in the backend temporarily
        trace_entry = {
            'traces': traces,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'heatmap': img_base64,
            'statistics': {
                'min': float(np.min(trace_array)),
                'max': float(np.max(trace_array)),
                'mean': float(np.mean(trace_array)),
                'std': float(np.std(trace_array)),
                'length': len(traces)
            }
        }
        
        stored_traces.append(trace_entry)
        
        # Also store just the heatmap info
        heatmap_entry = {
            'image': f'data:image/png;base64,{img_base64}',
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'statistics': trace_entry['statistics']
        }
        stored_heatmaps.append(heatmap_entry)
        
        # 4. Return the heatmap image and optionally other statistics to the frontend
        return jsonify({
            'success': True,
            'heatmap_url': f'data:image/png;base64,{img_base64}',
            'statistics': trace_entry['statistics'],
            'message': f'Trace collected successfully! Total traces: {len(stored_traces)}'
        })
        
    except Exception as e:
        print(f"Error in collect_trace: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implment a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        # 1. Clear stored traces and heatmaps
        global stored_traces, stored_heatmaps
        stored_traces.clear()
        stored_heatmaps.clear()
        
        # 2. Return success/error message
        return jsonify({
            'success': True,
            'message': 'All results cleared successfully!'
        })
        
    except Exception as e:
        print(f"Error in clear_results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_traces', methods=['GET'])
def get_traces():
    """Get all stored trace data for download"""
    try:
        traces_only = [trace['traces'] for trace in stored_traces]
        metadata = [trace['metadata'] for trace in stored_traces]
        
        return jsonify({
            'success': True,
            'traces': traces_only,
            'metadata': metadata,
            'count': len(traces_only)
        })
        
    except Exception as e:
        print(f"Error in get_traces: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_heatmaps', methods=['GET'])
def get_heatmaps():
    """Get all stored heatmaps"""
    try:
        return jsonify({
            'success': True,
            'heatmaps': stored_heatmaps,
            'count': len(stored_heatmaps)
        })
        
    except Exception as e:
        print(f"Error in get_heatmaps: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_results', methods=['GET'])
def get_results():
    """Get all stored results (used by collect.py)"""
    try:
        return jsonify({
            'success': True,
            'traces': [trace['traces'] for trace in stored_traces],
            'heatmaps': stored_heatmaps,
            'count': len(stored_traces)
        })
        
    except Exception as e:
        print(f"Error in get_results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)