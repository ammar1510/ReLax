"""ZMQ prompt sender for LLaMA inference.

This script sends dummy prompts over ZMQ to the inference consumer.

Usage:
    python prompt_sender.py
"""

import zmq
import time


def main():
    # Configuration
    num_workers = 4
    base_port = 5555

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # Connect to multiple ports (one per worker)
    worker_addresses = []
    for i in range(num_workers):
        port = base_port + i
        address = f"tcp://localhost:{port}"
        socket.connect(address)
        worker_addresses.append(address)
    
    print(f"Connected to {num_workers} workers:")
    for addr in worker_addresses:
        print(f"  - {addr}")

    # Dummy prompts to send
    prompts = [
        "The capital of France is",
        "In a galaxy far, far away",
        "The recipe for chocolate cake requires",
        "Python is a programming language that",
        "The meaning of life is",
        "Once upon a time in a distant land",
        "The best way to learn coding is",
        "Artificial intelligence will change",
        "The most important invention in history was",
        "If I could travel anywhere, I would go to",
        "The secret to happiness is",
        "In the future, technology will",
        "The greatest challenge facing humanity is",
        "My favorite thing about science is",
        "When I think about the universe, I",
        "The key to success is",
    ]

    print(f"\nSending {len(prompts)} prompts over ZMQ PUB socket...")
    print("Waiting for subscribers to connect...")
    time.sleep(3)  # Give subscribers time to connect (slow joiner problem)

    for i, prompt in enumerate(prompts):
        message = {"prompt": prompt, "request_id": f"request-{i}"}
        socket.send_json(message)
        print(f"Published prompt {i}: {prompt[:40]}...")
        time.sleep(0.01)  # Small delay to ensure delivery

    print(f"\nAll {len(prompts)} prompts published to {num_workers} workers!")
    socket.close()
    context.term()


if __name__ == "__main__":
    main()
