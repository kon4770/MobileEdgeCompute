"""
Run: python server.py
"""

import asyncio
import json
from collections import defaultdict
from websockets.server import serve
import numpy as np
import matplotlib.pyplot as plt
import datetime

REQUIRED_NODES = 1


BROKER_HOST = "192.168.0.113"
BROKER_PORT = 8765

# Configuration parameters (hardcoded for now; could be configurable later)
CONFIG = {
    "Nx": 1000,
    "Ny": 250,
    "Lx": 4.0,
    "Ly": 1.0 ,
    "A": 0.1,
    "eps": 0.25,
    "omega": 2 * np.pi / 10,
    "dt": 0.01,
    "total_time": 10.00,
    "nsteps": int(10.00 / 0.01)
}

class Broker:
    def __init__(self):
        self.nodes = {}
        self.conn_tid = {}
        self.ready_count = 0
        self.assign_next_tid = 0
        self.storage = {}
        self.waiters = defaultdict(list)
        self.final_data = {}
        self.completed_nodes = set()
        self.boundaries = []
        self.node_speeds = {}  # tid -> time per step
        self.has_started = False

    def compute_balanced_boundaries(self, num_nodes):
        # Compute column counts based on node speeds
        total_cols = CONFIG["Nx"]
        weights = [1.0 / max(speed, 1e-6) for speed in self.node_speeds.values()]
        print("weights:" + str(weights))
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        print("weights:" + str(weights))

        counts = [int(w * total_cols) for w in weights]
        print("counts:" + str(counts))
        remainder = total_cols - sum(counts)
        for i in range(remainder):
            counts[i] += 1

        starts = [0]
        for c in counts[:-1]:
            starts.append(starts[-1] + c)

        ends = [s + c for s, c in zip(starts, counts)]
        self.boundaries = list(zip(starts, ends))

        print("Balanced boundaries computed:" + str(self.boundaries))

    async def broadcast_start(self):
        msg = json.dumps({"type": "start"})
        for ws in self.nodes.values():
            await ws.send(msg)

    async def handler(self, ws):
        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")

                # -----------------------------------------------
                # Worker registration
                # -----------------------------------------------
                if t == "register_worker":

                    tid = self.assign_next_tid
                    self.assign_next_tid += 1
                    self.nodes[tid] = ws
                    self.conn_tid[ws] = tid
                    self.ready_count = len(self.nodes)

                    print(f"[server] Worker registered: tid={tid}")

                    if self.ready_count == REQUIRED_NODES:
                        print("[server] All workers connected. Measuring speeds...")
                        await self.measure_node_speeds()

                # -----------------------------------------------
                # Handle node speed measurement
                # -----------------------------------------------
                elif t == "speed_test_done":
                    print(f"[server] speed_test_done {data}")

                    tid = data["tid"]
                    speed = data["speed"]
                    self.node_speeds[tid] = speed

                    self.node_speeds = dict(sorted(self.node_speeds.items()))

                    print(f"[server] Speed of node {tid}: {speed}")
                    print(f"[server] if {self.node_speeds} : {len(self.node_speeds)} : {REQUIRED_NODES}")

                    if len(self.node_speeds) == REQUIRED_NODES:
                        print("[server] All speeds measured. Computing balanced boundaries...")
                        self.compute_balanced_boundaries(REQUIRED_NODES)
                        await self.send_balanced_config()

                elif t == "network_speed_test":
                    await ws.send(json.dumps({"type": "network_speed_test_return", "tid": data["tid"]}))


                # -----------------------------------------------
                # Publish boundary data
                # -----------------------------------------------
                elif t == "publish":
                    key = (data["tid"], data["step"], data["side"])
                    # print(f'{datetime.datetime.now()} [server] Worker publish data: {key}')
                    self.storage[key] = data["data"]
                    await ws.send(json.dumps({"type": "ack"}))

                    for waiter in self.waiters.pop(key, []):
                        # print(f'{datetime.datetime.now()} [server] Worker publish+send data: {key}')
                        await waiter.send(json.dumps({
                            "type": "response",
                            "found": True,
                            "tid": data["tid"],
                            "step": data["step"],
                            "side": data["side"],
                            "data": data["data"]
                        }))

                # -----------------------------------------------
                # Get boundary data
                # -----------------------------------------------
                elif t == "get":
                    key = (data["tid"], data["step"], data["side"])
                    # if data["side"] == "left":
                    #     print(f'{datetime.datetime.now()} [server] Worker {data["tid"]-1}requesting data: {key}')
                    # else:
                    #     print(f'{datetime.datetime.now()} [server] Worker {data["tid"]+1}requesting data: {key}')
                    if key in self.storage:
                        await ws.send(json.dumps({
                            "type": "response",
                            "found": True,
                            "tid": data["tid"],
                            "step": data["step"],
                            "side": data["side"],
                            "data": self.storage[key]
                        }))
                    else:
                        # print(f'{datetime.datetime.now()} [server] Worker needs to wait: {key}')
                        self.waiters[key].append(ws)

                # -----------------------------------------------
                # Handle final data (chunked)
                # -----------------------------------------------
                elif t == "final_data":
                    tid = data["tid"]
                    chunk = data.get("chunk")
                    total_chunks = data.get("total_chunks")

                    print(f"[server] Chunk {chunk} from {tid}")

                    if tid not in self.final_data:
                        self.final_data[tid] = [None] * total_chunks

                    self.final_data[tid][chunk] = data["data"]

                    # Send ACK back to the node
                    await ws.send(json.dumps({"type": "ack"}))

                    # Check if all chunks received
                    if all(self.final_data[tid]):
                        print(f"[server] All chunks received from node {tid}")
                        self.reconstruct_and_plot(tid)

        except Exception as e:
            print("Server exception:", e)
        finally:
            if ws in self.conn_tid:
                tid = self.conn_tid[ws]
                print(f"[server] Worker disconnected: {tid}")
                del self.conn_tid[ws]
                if tid in self.nodes:
                    del self.nodes[tid]

    async def measure_node_speeds(self):
        # Send a small test step to each node to measure speed
        for tid, ws in self.nodes.items():
            await ws.send(json.dumps({
                "type": "speed_test",
                "tid": tid,
                "steps": 500,  # Just a few steps to estimate speed
                "seed": np.random.randint(0, 2**8),
            }))

    async def send_balanced_config(self):
        for tid, ws in self.nodes.items():
            col0, col1 = self.boundaries[tid]
            config = {
                "type": "config",
                **CONFIG,
                "tid": tid,
                "col0": col0,
                "col1": col1,
                "total_nodes": REQUIRED_NODES
            }
            await ws.send(json.dumps(config))

        # Broadcast start signal after config sent
        await self.broadcast_start()

    def reconstruct_and_plot(self, tid):
        # Track how many nodes have completed
        if not hasattr(self, 'completed_nodes'):
            self.completed_nodes = set()

        self.completed_nodes.add(tid)

        # Check if all nodes have sent their data
        if len(self.completed_nodes) < REQUIRED_NODES:
            print(f"[server] Node {tid} finished. Waiting for others...")
            return

        print("[server] All nodes have sent final data. Reconstructing and plotting...")

        # Initialize full_data once
        if not hasattr(self, 'full_data'):
            self.full_data = np.zeros((CONFIG["Ny"], CONFIG["Nx"]))  # Ny x Nx

        # Reconstruct full data from all nodes
        for node_tid in range(REQUIRED_NODES):
            chunks = self.final_data[node_tid]
            data = np.concatenate(chunks, axis=1)

            col0, col1 = self.boundaries[node_tid]
            self.full_data[:, col0:col1] = data

        # Plot final result
        plt.figure(figsize=(12, 6))
        plt.imshow(self.full_data, extent=[0, CONFIG["Lx"], 0, CONFIG["Ly"]], origin='lower', cmap='viridis')
        plt.colorbar(label='Phi')
        plt.title('Final Phi Field (Combined from All Nodes)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig('final_phi_combined.png')
        print("[server] Combined plot saved as 'final_phi_combined.png'")


async def main():
    broker = Broker()
    print(f"Broker running on ws://{BROKER_HOST}:{BROKER_PORT}")
    async with serve(broker.handler, BROKER_HOST, BROKER_PORT):
        await asyncio.Future()    # run forever

if __name__ == "__main__":
    asyncio.run(main())
