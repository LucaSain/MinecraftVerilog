import random
import math
import redstone

class Optimizer:
    def __init__(self, factory, bounds=(50, 50), startT=1000, endT=0.1, cooling_rate=0.995):
        self.factory : redstone.RedstoneFactory = factory
        self.nodes = factory.graph.nodes
        self.bounds = bounds  # (Max X, Max Z) for random placement

        # Hyperparameters
        self.temp = startT
        self.cooling_rate = cooling_rate
        self.min_temp = endT

    def _random_pos(self):
        stride = 2
        x = random.randint(0, self.bounds[0]) * stride
        z = random.randint(0, self.bounds[1]) * stride
        return (x, 4, z)

    def propose_move(self):
        move_type = random.choice(['MOVE', 'MOVE', 'SWAP', 'ROTATE', 'VARIANT', 'REROUTE', 'REROUTE'])

        if move_type == 'MOVE':
            # UNLOCKED: Allowed inputs to move again
            node = random.choice(self.nodes)
            undo_data = ('MOVE', node, node.pos)
            node.pos = self._random_pos()
            return undo_data

        elif move_type == 'SWAP':
            # UNLOCKED: Allowed inputs to swap
            if len(self.nodes) < 2: return None
            n1, n2 = random.sample(self.nodes, 2)
            undo_data = ('SWAP', n1, n1.pos, n2, n2.pos)
            n1.pos, n2.pos = n2.pos, n1.pos
            return undo_data

        elif move_type == 'ROTATE':
            node = random.choice(self.nodes)
            undo_data = ('ROTATE', node, node.rotation)
            node.rotation = (node.rotation + 1) % 4
            return undo_data


        elif move_type == 'VARIANT':

            # FIX: Allow selecting variants for INPUTS (which have multiple OUTPUT options)

            candidates = []

            # 1. Standard Gates (Input Variants)

            candidates.extend([n for n in self.nodes if any(len(v) > 1 for v in n.proto.inputs.values())])

            # 2. Input Ports (Output Variants)

            candidates.extend([n for n in self.nodes if n.type == "INPUT"])

            if not candidates: return None

            node = random.choice(candidates)

            if node.type == "INPUT":

                # Handle Output Side Swapping for Inputs

                pin = 'Y'

                old_idx = node.selected_variants.get(pin, 0)

                undo_data = ('VARIANT', node, pin, old_idx)

                # Cycle through the 4 sides

                # We reuse set_variant logic, but we need to ensure get_global_socket reads it

                new_idx = (old_idx + 1) % 4

                node.set_variant(pin, new_idx)

                return undo_data

            else:

                # Handle Input Swapping for Gates (Existing Logic)

                complex_pins = [p for p, v in node.proto.inputs.items() if len(v) > 1]

                if not complex_pins: return None

                pin = random.choice(complex_pins)

                old_idx = node.selected_variants.get(pin, 0)

                undo_data = ('VARIANT', node, pin, old_idx)

                new_idx = (old_idx + 1) % len(node.proto.inputs[pin])

                node.set_variant(pin, new_idx)

                return undo_data

        elif move_type == 'REROUTE':
            # Pick a node that has outgoing edges
            candidates = [n for n in self.nodes if n.edges]
            if not candidates: return None

            node = random.choice(candidates)
            edge_idx = random.randint(0, len(node.edges) - 1)

            # Save old state: node.edges[edge_idx][3] is the boolean
            old_val = node.edges[edge_idx][3]
            undo_data = ('REROUTE', node, edge_idx, old_val)

            # Flip the boolean
            node.edges[edge_idx][3] = not old_val
            return undo_data

        return None

    def revert_move(self, undo_data):
        if not undo_data: return
        move_type = undo_data[0]

        if move_type == 'MOVE':
            _, node, old_pos = undo_data
            node.pos = old_pos
        elif move_type == 'SWAP':
            _, n1, p1, n2, p2 = undo_data
            n1.pos = p1
            n2.pos = p2
        elif move_type == 'ROTATE':
            _, node, old_rot = undo_data
            node.rotation = old_rot
        elif move_type == 'VARIANT':
            _, node, pin, old_idx = undo_data
            node.set_variant(pin, old_idx)
        elif move_type == 'REROUTE':
            _, node, idx, old_val = undo_data
            node.edges[idx][3] = old_val

    def run(self, steps=10000):
        # 1. Initial State
        current_energy = self.factory.graph.calculate_total_energy()
        best_energy = current_energy

        print(f"Starting SA. Initial Energy: {current_energy}")

        # Diagnostics
        accepted_moves = 0

        # 2. Optimization Loop
        for i in range(steps):
            # A. Propose a Move
            undo_data = self.propose_move()
            if not undo_data:
                continue

            # B. Evaluate
            new_energy = self.factory.graph.calculate_total_energy()
            delta = new_energy - current_energy

            # C. Acceptance Logic (Metropolis-Hastings)
            accept = False
            if delta < 0:
                accept = True  # Always accept improvements
            else:
                # Accept worse moves with probability exp(-delta / T)
                try:
                    prob = math.exp(-delta / self.temp)
                except OverflowError:
                    prob = 0

                if random.random() < prob:
                    accept = True

            # D. Apply or Revert
            if accept:
                current_energy = new_energy
                accepted_moves += 1

                # Update Global Best
                if new_energy < best_energy:
                    best_energy = new_energy
                    print(f"  >>> New Best at Step {i}: {best_energy}")
            else:
                self.revert_move(undo_data)

            # E. Heartbeat Status (Every 100 steps)
            if i % 100 == 0:
                print(f"Step {i}/{steps} | Temp: {self.temp:.2f} | Current: {current_energy} | Best: {best_energy}")

            # F. Cooling Schedule
            self.temp *= self.cooling_rate

            # G. Convergence Check
            if self.temp < self.min_temp:
                print(f"Converged at step {i} (Temp {self.temp:.2f} < {self.min_temp}).")
                break

        # 3. Final Report
        print(f"Finished. Final Energy: {current_energy}")
        print(f"Total Accepted Moves: {accepted_moves} / {steps}")
        return best_energy