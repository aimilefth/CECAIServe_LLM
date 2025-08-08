#!/usr/bin/python3
"""
Authors: Aimilios Leftheriotis, Achilleas Tzenetopoulos
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module implements a Flask server that provides endpoints for AI model inference and metric services.

Overview:
- The Flask server is configured to handle two primary services: 
  1. Inference Service
  2. Metric Service
- It utilizes the MyServer class from the my_server module to process the requests.
- The server is designed to be lightweight and efficient, ensuring minimal overhead during request handling.

Functionality:
1. Inference Service ('/api/infer'):
   - Accepts POST requests with input data for inference.
   - Enqueues the request for asynchronous processing.
   - Returns the encoded output from the MyServer instance as the response.

2. Metric Service ('/api/metrics'):
   - Accepts POST requests with parameters to fetch metrics.
   - Enqueues the request for asynchronous processing.
   - Returns either all metrics or a specified number of recent metrics based on the client's request.

Logging:
- The logging configuration is specified by the 'LOG_CONFIG' environment variable.
- The module sets up loggers for both file and console output.

Queue and Threading:
- A single queue is used to manage requests for both services.
- A worker thread processes the requests from the queue.
- Condition variables are used to synchronize request processing and result retrieval.

Usage:
- The server is started with the host and port specified by the 'SERVER_IP' and 'SERVER_PORT' environment variables.
- This implementation can serve as a template for building Flask servers for various machine learning inference and metric services.

Note:
- The server is designed to work with AI-framework/platform pair-specific server classes, which should be defined in their respective modules (e.g., {pair}_server.py).

This module can be extended and customized to support additional services or integrate with different AI models and platforms.
"""


# Import necessary libraries and modules
import os
import logging
import logging.config
import json
from flask import Flask, request, Response
import uuid
import queue
import threading
import my_server  # Import custom modules for the server's functionality and utility functions
import utils
import time
from werkzeug.serving import make_server

# Initialize Flask app instance
app = Flask(__name__)
# Single Queue and Condition for both services
request_queue = queue.Queue()
condition = threading.Condition()
results = {}
shutdown_event = threading.Event()  # <-- Introduce the shutdown event

# Worker function to process requests
def worker(logger):
    """
    Worker thread to process inference and metric requests.
    """
    time.sleep(0.1) # Small sleep to allow the app flask to start before the server initialization
    server = my_server.MyServer(logger)
    while True:
        # Wait for a request to be enqueued
        item = request_queue.get()
        service_identifier, request_id, request_dict = item
        if service_identifier == 'inference':
            encoded_output = server.inference(indata=request_dict['data'])
            result = server.send_response(encoded_output=encoded_output) 
        elif service_identifier == 'metric':
            json_input = request_dict['json']
            if utils.strtobool(json_input['all']):
                result = Response(response=json.dumps(server.get_metrics()), status=200, mimetype='application/json')
            else:
                number = json_input['number']
                if isinstance(number, int) and number > 0:
                    result = Response(response=json.dumps(server.get_metrics(number)), status=200, mimetype='application/json')
                else:
                    result = Response(response={'number': 'invalid'}, status=400, mimetype='application/json')
        elif service_identifier == 'power':
             # Do server.calculate_power()
            result = Response(response=json.dumps(server.get_power_metrics()), status=200, mimetype='application/json')
        elif service_identifier == 'logs':
             # Do server.calculate_power()
            result = Response(response=server.get_logs(), status=200, mimetype='text/plain')
        elif service_identifier == 'shutdown':
            result = Response(response='Server is shutting down...', status=200, mimetype='text/plain')
            with condition:
                results[request_id] = result
                condition.notify_all()
            break
        else:
            raise AssertionError(f'This should not happen, got service_identifier {service_identifier}')

        # Store the result and notify
        with condition:
            results[request_id] = result
            condition.notify_all()

@app.route('/api/infer', methods=['POST'])
def inference_service():
    """
    Service for performing inference on data received in POST requests.
    Enqueue the request data and return a response immediately.
    """
    request_id = str(uuid.uuid4())
    request_dict = {'data': request.data}
    # Enqueue the request for processing
    with condition:
        request_queue.put(('inference', request_id, request_dict))
        condition.wait_for(lambda: request_id in results)
    # Get the result and return
    result = results.pop(request_id)
    return result

@app.route('/api/metrics', methods=['POST'])
def metric_service():
    """
    Service for fetching metrics based on the parameters received in the POST request.
    Enqueue the request data and return a response immediately.
    """
    json_input = request.get_json()
    request_id = str(uuid.uuid4())
    request_dict = {'json': json_input}
    # Enqueue the request for processing
    with condition:
        request_queue.put(('metric', request_id, request_dict))
        condition.wait_for(lambda: request_id in results)
    # Get the result and return
    result = results.pop(request_id)
    return result

@app.route('/api/power', methods=['POST'])
def power_service():
    """
    Service for performing power monitoring
    Enqueue the request data and return a response immediately.
    """
    json_input = request.get_json()
    request_id = str(uuid.uuid4())
    request_dict = {'data': 'placeholder'}
    # Enqueue the request for processing
    with condition:
        request_queue.put(('power', request_id, request_dict))
        condition.wait_for(lambda: request_id in results)
    # Get the result and return
    result = results.pop(request_id)
    return result

@app.route('/api/logs', methods=['POST'])
def logs_service():
    """
    Endpoint to retrieve server logs.
    """
    request_id = str(uuid.uuid4())
    request_dict = {'data': 'placeholder'}
    # Enqueue the request for processing
    with condition:
        request_queue.put(('logs', request_id, request_dict))
        condition.wait_for(lambda: request_id in results)
    # Get the result and return
    result = results.pop(request_id)
    return result

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """
    Service to shut down the server and worker thread.
    """   
    request_id = str(uuid.uuid4())
    request_dict = {'data': 'placeholder'}
    # Enqueue the request for processing
    with condition:
        request_queue.put(('shutdown', request_id, request_dict))
        condition.wait_for(lambda: request_id in results)
    # Get the result and return
    result = results.pop(request_id)
    # Register the shutdown function to be called after the response is sent
    result.call_on_close(shutdown_event.set)
    return result

def main():
    """
    Main function to configure logging, start the worker thread, and run the Flask app.
    """
    # Configure logging based on the environmental variable 'LOG_CONFIG'
    logging.config.fileConfig(os.environ['LOG_CONFIG'], disable_existing_loggers=False)

    # Create logger instances for both logging to file and console
    logger = logging.getLogger('sampleLogger')  # Logger for logging to file
    root_logger = logging.getLogger()  # Logger for logging to console

    # Start the worker thread
    worker_thread = threading.Thread(target=worker, args=(logger,), daemon=True)
    worker_thread.start()

    # Run the Flask app in a separate thread
    server = make_server(os.environ['SERVER_IP'], int(os.environ['SERVER_PORT']), app)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    
     # Print the server startup messages
    host = os.environ['SERVER_IP']
    port = int(os.environ['SERVER_PORT'])
    print(f" * Running on http://{host}:{port}")
    # Get all non-loopback IP addresses
    ip_addresses = utils.get_local_ip_addresses()
    for ip in ip_addresses:
        print(f" * Running on http://{ip}:{port}")
    print("Press CTRL+C x2 to quit")

    logger.info(f" * Running on http://{host}:{port}")
    for ip in ip_addresses:
        logger.info(f" * Running on http://{ip}:{port}")
    logger.info("Press CTRL+C x2 to quit")

    # Wait for the shutdown event
    shutdown_event.wait()

    # Shutdown the server
    server.shutdown()
    worker_thread.join()
    server_thread.join()
     
if __name__ == '__main__':
    main()

