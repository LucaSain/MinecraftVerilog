import heapq
import os
import re
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
# --- AMULET IMPORTS ---
from amulet.api.block import Block
from amulet.api.data_types import BlockCoordinates
from amulet.api.selection import SelectionGroup, SelectionBox
from amulet.api.level import Structure
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper

def rotate_vector(x: int, z: int, rotation: int) -> Tuple[int, int]:
    """Rotates a vector (position or direction) by 90 deg steps."""
    rotation = rotation % 4
    if rotation == 0:
        return x, z
    elif rotation == 1:
        return -z, x
    elif rotation == 2:
        return -x, -z
    elif rotation == 3:
        return z, -x
    return x, z


def rotate_blockstate(block_str: str, rotation: int) -> str:
    rotation = rotation % 4
    if rotation == 0: return block_str
    dirs = ["north", "east", "south", "west"]
    match = re.search(r"facing=([a-z]+)", block_str)
    if match:
        current_facing = match.group(1)
        if current_facing in dirs:
            idx = dirs.index(current_facing)
            new_idx = (idx + rotation) % 4
            new_facing = dirs[new_idx]
            return block_str.replace(f"facing={current_facing}", f"facing={new_facing}")
    return block_str


# ==========================================
# 2. AMULET SCHEMATIC WRAPPER (FIXED SAVE)
# ==========================================

class AmuletSchematic:
    def __init__(self):
        self._blocks: Dict[BlockCoordinates, Block] = {}
        self.min_coords = [float('inf'), float('inf'), float('inf')]
        self.max_coords = [float('-inf'), float('-inf'), float('-inf')]

    def wire_up(self,a,b,c):
        return
        #skip

    def wire_down(self,a,b,c):
        return
        #skip

    def is_empty(self, coords: BlockCoordinates) -> bool:
        return coords not in self._blocks
    def setBlock(self, coords: BlockCoordinates, block_str: str):
        x, y, z = coords
        self.min_coords[0] = min(self.min_coords[0], x)
        self.min_coords[1] = min(self.min_coords[1], y)
        self.min_coords[2] = min(self.min_coords[2], z)
        self.max_coords[0] = max(self.max_coords[0], x)
        self.max_coords[1] = max(self.max_coords[1], y)
        self.max_coords[2] = max(self.max_coords[2], z)
        self._blocks[(x, y, z)] = Block.from_string_blockstate(block_str)
        if block_str == "minecraft:copper_bulb":
            print(Block.from_string_blockstate("minecraft:copper_block"))

    def save(self, folder_path, file_name):
        if not self._blocks: return
        if not os.path.exists(folder_path): os.makedirs(folder_path)

        # 1. Use .schem extension (critical for 1.21 support)
        full_path = os.path.join(folder_path, f"{file_name}.schem")

        # 2. Calculate Bounds
        width = int(self.max_coords[0] - self.min_coords[0]) + 1
        height = int(self.max_coords[1] - self.min_coords[1]) + 1
        length = int(self.max_coords[2] - self.min_coords[2]) + 1
        ox, oy, oz = int(self.min_coords[0]), int(self.min_coords[1]), int(self.min_coords[2])

        selection = SelectionGroup([SelectionBox((0, 0, 0), (width, height, length))])

        # 3. INITIALIZE FILE (The Fix)
        # We create the file headers and immediately close the wrapper.
        wrapper = SpongeSchemFormatWrapper(full_path)
        wrapper.create_and_open("java", (1, 21, 0), bounds=selection, overwrite=True)
        wrapper.save()
        wrapper.close()

        # 4. RE-OPEN WITH STRUCTURE
        # Now we create a fresh wrapper instance for the Structure class to use.
        # Structure() calls .open() internally, so this wrapper must be closed initially.
        wrapper_reopen = SpongeSchemFormatWrapper(full_path)
        struct = Structure(full_path, wrapper_reopen)

        dim = struct.dimensions[0]  # Usually 'main'

        # 5. Place Blocks
        for (bx, by, bz), block in self._blocks.items():
            lx, ly, lz = int(bx - ox), int(by - oy), int(bz - oz)
            struct.set_version_block(lx, ly, lz, dim, ("java", (1,21, 4)), block)

        struct.save()
        struct.close()
        print(f"Saved to: {full_path}")


# ==========================================
# 3. GEOMETRY ENGINE
# ==========================================

class BoundingBox:
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x, self.max_x = sorted((min_x, max_x))
        self.min_y, self.max_y = sorted((min_y, max_y))
        self.min_z, self.max_z = sorted((min_z, max_z))

    def intersects(self, other: 'BoundingBox') -> bool:
        return (self.min_x < other.max_x and self.max_x > other.min_x and
                self.min_y < other.max_y and self.max_y > other.min_y and
                self.min_z < other.max_z and self.max_z > other.min_z)


# ==========================================
# 4. COMPONENT DEFINITIONS
# ==========================================

class Component:
    def __init__(self, width, height, length):
        self.size = (width, height, length)
        self.inputs = {}
        self.outputs = {}

    def is_within_local_bounds(self, x: int, y: int, z: int) -> bool:
        """
        Checks if a local coordinate (relative to the component's origin anchor)
        is strictly inside the component's defined bounding box.
        """
        w, h, l = self.size
        return 0 <= x < w and 0 <= y < h and 0 <= z < l

    def place(self, schem, x, y, z, rotation=0, debug=False):
        raise NotImplementedError()


class ComponentInstance:
    def __init__(self, uid: int, prototype: Component, pos: Tuple[int, int, int], type_name: str):
        self.id = uid
        self.proto = prototype
        self.type = type_name
        self.pos = pos
        self.rotation = 0
        self.selected_variants: Dict[str, int] = {}
        self.edges = []
        for input_name in self.proto.inputs: self.selected_variants[input_name] = 0

    def get_bounding_box(self) -> BoundingBox:
        # 1. Get dimensions
        w, h, l = self.proto.size

        # 2. Identify the 4 integer corners of the footprint (inclusive)
        #    We use w-1 and l-1 because a width of 1 occupies index 0 only.
        corners = [
            (0, 0),
            (w - 1, 0),
            (0, l - 1),
            (w - 1, l - 1)
        ]

        # 3. Rotate these integer coordinates
        rotated_corners = []
        for (lx, lz) in corners:
            rx, rz = rotate_vector(lx, lz, self.rotation)
            rotated_corners.append((rx, rz))

        # 4. Find min/max in local rotated space
        min_rx = min(c[0] for c in rotated_corners)
        max_rx = max(c[0] for c in rotated_corners)
        min_rz = min(c[1] for c in rotated_corners)
        max_rz = max(c[1] for c in rotated_corners)

        # 5. Calculate global bounds
        #    Note: We add +1 to the max values to create the "exclusive" bound
        #    required for Python slicing (grid[min:max]) and bounding box intersection.
        gx, gy, gz = self.pos
        return BoundingBox(
            gx + min_rx, gy, gz + min_rz,
            gx + max_rx + 1, gy + h, gz + max_rz + 1
        )

    def get_global_socket(self, pin_name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        if pin_name in self.proto.inputs:
            idx = self.selected_variants[pin_name]
            if idx >= len(self.proto.inputs[pin_name]): idx = 0
            rx, ry, rz, nx, ny, nz = self.proto.inputs[pin_name][idx]
        elif pin_name in self.proto.outputs:
            idx = 0
            if len(self.proto.outputs[pin_name]) > 1:
                idx = self.selected_variants.get(pin_name, 0)
            rx, ry, rz, nx, ny, nz = self.proto.outputs[pin_name][idx]
        else:
            raise ValueError(f"Pin {pin_name} not found on {self.type}")

        rot_x, rot_z = rotate_vector(rx, rz, self.rotation)
        rot_nx, rot_nz = rotate_vector(nx, nz, self.rotation)
        gx, gy, gz = self.pos
        return (gx + rot_x, gy + ry, gz + rot_z), (rot_nx, ny, rot_nz)

    def connect(self, other, my_pin='Y', other_pin='A'):
        self.edges.append([other, my_pin, other_pin, False])

    def set_variant(self, pin_name: str, variant_index: int):
        self.selected_variants[pin_name] = variant_index

    def __repr__(self):
        return f"[{self.type} #{self.id}]"


class RotatableComponent(Component):
    def _place_block(self, schem, anchor_x, anchor_y, anchor_z, rel_x, rel_y, rel_z, block_str, rotation):
        rot_x, rot_z = rotate_vector(rel_x, rel_z, rotation)
        final_block = rotate_blockstate(block_str, rotation)
        schem.setBlock((anchor_x + rot_x, anchor_y + rel_y, anchor_z + rot_z), final_block)

    def place_foundation(self, schem, x, y, z, rotation):
        width, height, length = self.size
        for ix in range(width):
            for iz in range(length):
                rot_x, rot_z = rotate_vector(ix, iz, rotation)
                schem.setBlock((x + rot_x, y - 1, z + rot_z), "minecraft:stone")

    def draw_debug_bounds(self, schem, x, y, z, rotation):
        """
        Renders a platform of iron blocks above the component to visualize
        the XZ plane projection of its bounding box.
        """
        width, height, length = self.size
        # Draw 1 block above the component's height
        debug_y = y + height

        for ix in range(width):
            for iz in range(length):
                # Rotate the local footprint coordinate to match world placement
                rot_x, rot_z = rotate_vector(ix, iz, rotation)
                schem.setBlock((x + rot_x, debug_y, z + rot_z), "minecraft:iron_block")


# --- UPDATED CONCRETE COMPONENTS ---

class InputPort(RotatableComponent):
    def __init__(self):
        super().__init__(1, 1, 1)
        self.inputs = {}
        self.outputs = {
            'Y': [(0, 0, 1, 0, 0, 1), (0, 0, -1, 0, 0, -1), (1, 0, 0, 1, 0, 0), (-1, 0, 0, -1, 0, 0)]
        }

    def place(self, schem, x, y, z, rotation=0, debug=False):
        self.place_foundation(schem, x, y, z, rotation)
        schem.setBlock((x, y, z), "minecraft:lapis_block")
        schem.setBlock((x, y + 1, z), "minecraft:lever")
        if debug: self.draw_debug_bounds(schem, x, y, z, rotation)


class NorGate(RotatableComponent):
    def __init__(self):
        super().__init__(3, 2, 3)
        self.inputs = {'A': [(3, 0, 1, 1, 0, 0)], 'B': [(-1, 0, 1, -1, 0, 0), (1, 0, -1, 0, 0, -1)]}
        self.outputs = {'Y': [(1, 0, 4, 0, 0, 1)]}

    def place(self, schem, x, y, z, rotation=0, debug=False):
        self.place_foundation(schem, x, y, z, rotation)
        blocks = [(1, 0, 1, "minecraft:gold_block"),
                  (1, 0, 2, "minecraft:redstone_wall_torch[facing=south]"), (1, 0, 3, "minecraft:redstone_wire"),
                  (2, 0, 1, "minecraft:redstone_wire"), (1, 0, 0, "minecraft:redstone_wire"),
                  (0, 0, 1, "minecraft:redstone_wire")]
        for rx, ry, rz, b in blocks: self._place_block(schem, x, y, z, rx, ry, rz, b, rotation)
        if debug: self.draw_debug_bounds(schem, x, y, z, rotation)


class DFlipFlop(RotatableComponent):
    def __init__(self):
        # Reduced width from 5 to 4
        super().__init__(4, 2, 5)

        # Shifted all Input/Output X coordinates by -1
        self.inputs = {
            'C': [(-1, -2, 2, -1, 0, 0)],
            'D': [(2, 0, 5, 0, 0, 1), (4, 0, 4, 1, 0, 0)]
        }
        self.outputs = {
            'Q': [(2, 0, -1, 0, 0, -1), (4, 0, 1, 1, 0, 0)]
        }

    def place(self, schem, x, y, z, rotation=0, debug=False):
        self.place_foundation(schem, x, y, z, rotation)

        # Shifted all block X coordinates by -1
        structure = [
            (2, 0, 4, "minecraft:lime_wool"),
            (-1, -2, 2, "minecraft:redstone_wire"),
            (2, 0, 5, "minecraft:redstone_wire"),
            (3, 0, 4, "minecraft:redstone_wire"),
            (0, 0, 3, "minecraft:redstone_torch"),
            (0, -2, 3, "minecraft:redstone_wall_torch[facing=south]"),
            (1, 0, 3, "minecraft:repeater[facing=west]"),
            (1, 0, 2, "minecraft:repeater[facing=west]"),
            (2, 0, 3, "minecraft:repeater[facing=south]"),
            (2, 0, 2, "minecraft:repeater[facing=south]"),
            (2, 0, 1, "minecraft:pink_wool"),
            (2, 0, 0, "minecraft:redstone_wire"),
            (3, 0, 1, "minecraft:redstone_wire")
        ]

        for rx, ry, rz, b in structure:
            self._place_block(schem, x, y, z, rx, ry, rz, b, rotation)

        if debug:
            self.draw_debug_bounds(schem, x, y, z, rotation)


class NotGate(RotatableComponent):
    def __init__(self):
        super().__init__(3, 2, 3)
        self.inputs = {'A': [(3, 0, 1, 1, 0, 0), (-1, 0, 1, -1, 0, 0), (1, 0, -1, 0, 0, -1)]}
        self.outputs = {'Y': [(1, 0, 4, 0, 0, 1)]}

    def place(self, schem, x, y, z, rotation=0, debug=False):
        self.place_foundation(schem, x, y, z, rotation)
        blocks = [(1, 0, 1, "minecraft:gold_block"),
                  (1, 0, 2, "minecraft:redstone_wall_torch[facing=south]"), (1, 0, 3, "minecraft:redstone_wire"),
                  (2, 0, 1, "minecraft:redstone_wire"), (1, 0, 0, "minecraft:redstone_wire"),
                  (0, 0, 1, "minecraft:redstone_wire")]
        for rx, ry, rz, b in blocks: self._place_block(schem, x, y, z, rx, ry, rz, b, rotation)
        if debug: self.draw_debug_bounds(schem, x, y, z, rotation)


class Wire(RotatableComponent):
    def __init__(self):
        super().__init__(1, 1, 1)
        self.inputs = {'A': [(0, 0, 0, 0, 0, 0)]}
        self.outputs = {'Y': [(0, 0, 0, 0, 0, 0)]}

    def place(self, schem, x, y, z, rotation=0):
        self.place_foundation(schem, x, y, z, rotation)
        self._place_block(schem, x, y, z, 0, 0, 0, "minecraft:redstone_wire", rotation)


# ==========================================
# 5. GRAPH & FACTORY
# ==========================================

class CircuitGraph:
    def __init__(self):
        self.nodes: List[ComponentInstance] = []
        self._id_counter = 0

    def add_node(self, prototype: Component, pos: Tuple[int, int, int], type_name: str) -> ComponentInstance:
        node = ComponentInstance(self._id_counter, prototype, pos, type_name)
        self.nodes.append(node)
        self._id_counter += 1
        return node

    def calculate_total_energy(self) -> float:
        """
        Calculates energy based on:
        1. Component Overlap (Critical Penalty)
        2. Total Wire Length (Manhattan Distance)
        3. Wire Crossing Probability (Bounding Box Overlap Penalty)
        """
        cost = 0
        PENALTY_COMP_OVERLAP = 100000
        PENALTY_WIRE_CROSSING = 1000

        n = len(self.nodes)

        # 1. Component Overlaps
        for i in range(n):
            bbox_a = self.nodes[i].get_bounding_box()
            for j in range(i + 1, n):
                bbox_b = self.nodes[j].get_bounding_box()
                if bbox_a.intersects(bbox_b):
                    cost += PENALTY_COMP_OVERLAP

        # 2. Wire Analysis (Length + Intersections)
        wires = []  # Stores tuple: (min_x, min_z, max_x, max_z)

        for node in self.nodes:
            for edge in node.edges:
                # Edge structure: [target_node, src_pin, tgt_pin, is_inverted]
                target_node, src_pin, tgt_pin, _ = edge

                # Get Global Positions
                # We use a simplified center-to-center or pin-to-pin calculation
                # (Pin-to-pin is better for accuracy)
                p1, _ = node.get_global_socket(src_pin)
                p2, _ = target_node.get_global_socket(tgt_pin)

                x1, z1 = p1[0], p1[2]
                x2, z2 = p2[0], p2[2]

                # A. Manhattan Distance Cost
                dist = abs(x1 - x2) + abs(z1 - z2)
                cost += dist

                # Store Bounding Box of the wire for intersection check
                # Format: (min_x, min_z, max_x, max_z)
                w_bbox = (min(x1, x2), min(z1, z2), max(x1, x2), max(z1, z2))
                wires.append(w_bbox)

        # 3. Wire Crossing Penalty
        # We check if the "bounding box" of wire A overlaps wire B.
        # This is a standard heuristic for routing congestion.
        num_wires = len(wires)
        for i in range(num_wires):
            wx1, wz1, wx2, wz2 = wires[i]
            for j in range(i + 1, num_wires):
                ox1, oz1, ox2, oz2 = wires[j]

                # Check for Rectangle Intersection
                # (RectA Left < RectB Right) and (RectA Right > RectB Left) ...
                if (wx1 < ox2 and wx2 > ox1 and wz1 < oz2 and wz2 > oz1):
                    cost += PENALTY_WIRE_CROSSING

        return cost


class RedstoneFactory:
    def __init__(self):
        self.graph = CircuitGraph()
        self._prototypes = {
            "NOR": NorGate(), "NOT": NotGate(), "DFF": DFlipFlop(), "WIRE": Wire(), "INPUT": InputPort()
        }

    def create(self, type_name: str, x: int, y: int, z: int) -> ComponentInstance:
        if type_name not in self._prototypes: raise ValueError(f"Unknown: {type_name}")
        return self.graph.add_node(self._prototypes[type_name], (x, y, z), type_name)

    def compile_to_schematic(self, schem_wrapper: AmuletSchematic):
        print(f"Compiling {len(self.graph.nodes)} components...")

        BASE_Y = 4
        # 6 Layers for vertical freedom
        LAYER_HEIGHTS = [4, 7, 10, 13, 16, 19]
        NUM_LAYERS = len(LAYER_HEIGHTS)

        # --- WIRE COLORS ---
        WIRE_COLORS = [
            "white", "orange", "magenta", "light_blue", "yellow", "lime",
            "pink", "gray", "light_gray", "cyan", "purple", "blue",
            "brown", "green", "red", "black"
        ]

        # 1. PLACE COMPONENTS
        all_x, all_z = [], []
        for node in self.graph.nodes:
            node.pos = (node.pos[0], BASE_Y, node.pos[2])
            node.proto.place(schem_wrapper, node.pos[0], node.pos[1], node.pos[2], rotation=node.rotation)
            bbox = node.get_bounding_box()
            all_x.extend([bbox.min_x, bbox.max_x])
            all_z.extend([bbox.min_z, bbox.max_z])

        if not all_x: all_x, all_z = [0], [0]
        min_x, max_x = min(all_x) - 10, max(all_x) + 10
        min_z, max_z = min(all_z) - 10, max(all_z) + 10
        width = max_x - min_x + 1
        length = max_z - min_z + 1

        # 2. GENERATE SOLID FOUNDATION
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                schem_wrapper.setBlock((x, 0, z), "minecraft:bedrock")
                for y_fill in range(1, 4):
                    if schem_wrapper.is_empty((x, y_fill, z)):
                        schem_wrapper.setBlock((x, y_fill, z), "minecraft:stone")

        # 3. OBSTACLE GRID (With Increased Gate Padding)
        print("Calculating 3D Obstacle Grid...")
        grid_3d = np.zeros((width, NUM_LAYERS, length), dtype=int)

        # Increased padding to prevent artifacts near gates
        COMPONENT_PADDING = 6

        for node in self.graph.nodes:
            bbox = node.get_bounding_box()
            # Calculate padded bounds (clamped to grid size)
            sx = max(0, bbox.min_x - min_x - COMPONENT_PADDING)
            ex = min(width, bbox.max_x - min_x + COMPONENT_PADDING)
            sz = max(0, bbox.min_z - min_z - COMPONENT_PADDING)
            ez = min(length, bbox.max_z - min_z + COMPONENT_PADDING)

            # Block Layers 0, 1, and 2 around components to create a "Keep Out" zone
            grid_3d[sx:ex, 0:3, sz:ez] = 1

        # ==========================================
        # 3.5 PRE-RESERVE VERTICAL SOCKETS
        # ==========================================
        print("Reserving Vertical Socket Channels...")
        active_socket_locs = set()
        for node in self.graph.nodes:
            for edge in node.edges:
                target_node, src_pin_name, tgt_pin_name, _ = edge
                start_pos, _ = node.get_global_socket(src_pin_name)
                active_socket_locs.add((start_pos[0], start_pos[2]))
                end_pos, _ = target_node.get_global_socket(tgt_pin_name)
                active_socket_locs.add((end_pos[0], end_pos[2]))

        for (sx, sz) in active_socket_locs:
            gx, gz = sx - min_x, sz - min_z
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, nz = gx + dx, gz + dz
                    if 0 <= nx < width and 0 <= nz < length:
                        grid_3d[nx, :, nz] = 1

        router = AStarRouter()

        # ==========================================
        # 5. TRACE UNDERGROUND CLOCK (Restored)
        # ==========================================
        print("Tracing underground clock...")
        clock_grid = np.zeros((width, NUM_LAYERS, length), dtype=int)
        for node in self.graph.nodes:
            if node.type == "DFF":
                bbox = node.get_bounding_box()
                sx = max(0, bbox.min_x - min_x)
                ex = min(width, bbox.max_x - min_x)
                sz = max(0, bbox.min_z - min_z)
                ez = min(length, bbox.max_z - min_z)
                clock_grid[sx:ex, :, sz:ez] = 1

        clock_bus_z_index = length - 2
        clock_bus_z_world = min_z + clock_bus_z_index

        bus_signal_dist = 0
        for x_bus in range(min_x, max_x + 1):
            bus_signal_dist += 1
            if bus_signal_dist >= 10:
                schem_wrapper.setBlock((x_bus, 2, clock_bus_z_world), "minecraft:repeater[facing=east]")
                bus_signal_dist = 0
            else:
                schem_wrapper.setBlock((x_bus, 2, clock_bus_z_world), "minecraft:redstone_wire")

        routed_paths = []
        for node in self.graph.nodes:
            if 'C' in node.proto.inputs:
                start_pos, start_socket_def = node.get_global_socket('C')
                start_tuple = (start_pos[0] - min_x, 0, start_pos[2] - min_z)
                end_tuple = (start_tuple[0], 0, clock_bus_z_index)

                if 0 <= start_tuple[0] < width and 0 <= start_tuple[2] < length:
                    clock_grid[start_tuple[0], 0, start_tuple[2]] = 0
                if 0 <= end_tuple[0] < width and 0 <= end_tuple[2] < length:
                    clock_grid[end_tuple[0], 0, end_tuple[2]] = 0

                path_indices = router.find_path(clock_grid, start_tuple, end_tuple, start_socket_def, None)
                if path_indices:
                    for (px, layer, pz) in path_indices:
                        clock_grid[px, layer, pz] = 1
                        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, nz = px + dx, pz + dz
                            if 0 <= nx < width and 0 <= nz < length:
                                clock_grid[nx, layer, nz] = 1

                    mandatory_indices = set()
                    signal_dist = 0
                    for i in range(len(path_indices)):
                        signal_dist += 1
                        if signal_dist >= 7 and 0 < i < len(path_indices) - 1:
                            prev_x, _, prev_z = path_indices[i - 1]
                            curr_x, _, curr_z = path_indices[i]
                            next_x, _, next_z = path_indices[i + 1]
                            if not ((curr_x - prev_x != next_x - curr_x) or (curr_z - prev_z != next_z - curr_z)):
                                mandatory_indices.add(i)
                                signal_dist = 0

                    routed_paths.append({'path': path_indices, 'mandatory': mandatory_indices, 'start': start_pos})
                else:
                    print(f"Failed to route clock for {node.type} #{node.id}")

        if routed_paths:
            max_repeater_count = max(len(p['mandatory']) for p in routed_paths)
            for item in routed_paths:
                delay_needed = max_repeater_count - len(item['mandatory'])
                for i, (gx, layer, gz) in enumerate(item['path']):
                    wx, wz = gx + min_x, gz + min_z
                    block_type = "minecraft:redstone_wire"
                    facing = "north"
                    if i < len(item['path']) - 1:
                        nx, _, nz = item['path'][i + 1]
                        if nz - gz == 1:
                            facing = "south"
                        elif nz - gz == -1:
                            facing = "north"
                        elif nx - gx == 1:
                            facing = "east"
                        elif nx - gx == -1:
                            facing = "west"

                    if i in item['mandatory']:
                        block_type = f"minecraft:repeater[facing={facing}]"
                    elif delay_needed > 0 and 0 < i < len(item['path']) - 1:
                        prev_x, _, prev_z = item['path'][i - 1]
                        next_x, _, next_z = item['path'][i + 1]
                        if not ((gx - prev_x != next_x - gx) or (gz - prev_z != next_z - gz)):
                            block_type = f"minecraft:repeater[facing={facing}]";
                            delay_needed -= 1

                    schem_wrapper.setBlock((wx, 2, wz), block_type)
                    if wx == item['start'][0] and wz == item['start'][2]:
                        schem_wrapper.setBlock((wx, 3, wz), "minecraft:glass")
                        schem_wrapper.setBlock((wx, 3, wz), "minecraft:redstone_wire")

        # ==========================================
        # 6. ROUTE LOGIC SIGNALS
        # ==========================================
        print("Routing Logic Signals (3D Mode with Fan-Out & Observer Towers)...")

        wire_counter = 0
        pin_usage_tracker = {}

        for node in self.graph.nodes:
            for edge in node.edges:
                target_node, src_pin_name, tgt_pin_name, _ = edge

                # --- SKIP CLOCK ROUTING ---
                if tgt_pin_name == 'C' and target_node.type == 'DFF':
                    continue

                current_color = WIRE_COLORS[wire_counter % len(WIRE_COLORS)]
                wool_block = f"minecraft:{current_color}_wool"
                wire_counter += 1

                # 1. Get Base Coordinates
                start_pos, start_socket_def = node.get_global_socket(src_pin_name)
                end_pos, end_socket_def = target_node.get_global_socket(tgt_pin_name)

                # --- FAN-OUT BUS LOGIC ---
                socket_key = (node.id, src_pin_name)
                usage_count = pin_usage_tracker.get(socket_key, 0)
                pin_usage_tracker[socket_key] = usage_count + 1

                start_nx, start_nz = 0, 0
                if start_socket_def:
                    start_nx, start_nz = start_socket_def[0], start_socket_def[2]

                shifted_start_x = start_pos[0] + (start_nx * usage_count)
                shifted_start_z = start_pos[2] + (start_nz * usage_count)

                # Draw Bus
                if usage_count > 0:
                    for k in range(usage_count + 1):
                        bus_x = start_pos[0] + (start_nx * k)
                        bus_z = start_pos[2] + (start_nz * k)
                        schem_wrapper.setBlock((bus_x, BASE_Y - 1, bus_z), wool_block)
                        schem_wrapper.setBlock((bus_x, BASE_Y, bus_z), "minecraft:redstone_wire")

                        # Mark Bus in Grid
                        bgx, bgz = bus_x - min_x, bus_z - min_z
                        if 0 <= bgx < width and 0 <= bgz < length:
                            for layer_idx in [0, 1]:
                                for px in [-1, 0, 1]:
                                    for pz in [-1, 0, 1]:
                                        if 0 <= bgx + px < width and 0 <= bgz + pz < length:
                                            grid_3d[bgx + px, layer_idx, bgz + pz] = 1

                # --- TARGET OFFSET ---
                target_offset_x, target_offset_z = 0, 0
                if end_socket_def:
                    target_offset_x, target_offset_z = end_socket_def[0], end_socket_def[2]

                start_tuple = (shifted_start_x - min_x, 0, shifted_start_z - min_z)
                end_tuple = (
                    (end_pos[0] + target_offset_x) - min_x,
                    0,
                    (end_pos[2] + target_offset_z) - min_z
                )

                # 2. Prepare Grid & UNLOCK Reserved Spots
                current_grid = grid_3d.copy()

                def unlock_reserved_socket(gx, gz):
                    if not (0 <= gx < width and 0 <= gz < length): return
                    current_grid[gx, :, gz] = 0
                    for dx in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nx, nz = gx + dx, gz + dz
                            if 0 <= nx < width and 0 <= nz < length:
                                current_grid[nx, :, nz] = 0

                unlock_reserved_socket(start_tuple[0], start_tuple[2])
                unlock_reserved_socket(end_tuple[0], end_tuple[2])

                # 3. Find 3D Path
                path_indices = router.find_path(current_grid, start_tuple, end_tuple, start_socket_def, end_socket_def)

                if path_indices:
                    signal_dist = 0
                    pending_comparator = False

                    for i in range(len(path_indices)):
                        gx, layer, gz = path_indices[i]
                        wx, wz, wy = gx + min_x, gz + min_z, LAYER_HEIGHTS[layer]

                        # Update Global Grid (Block current position)
                        grid_3d[gx, layer, gz] = 1
                        # 4-neighbor padding for horizontal wires
                        for px, pz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if 0 <= gx + px < width and 0 <= gz + pz < length:
                                grid_3d[gx + px, layer, gz + pz] = 1

                        # --- PLACEMENT LOGIC ---

                        # 1. Vertical Transition (Observer Tower)
                        prev_layer = path_indices[i - 1][1] if i > 0 else layer

                        if prev_layer != layer:
                            # 1. CHECK CONTINUITY
                            is_continuous_vertical = False
                            if i < len(path_indices) - 1:
                                n_gx, n_layer, n_gz = path_indices[i + 1]
                                if n_gx == gx and n_gz == gz:
                                    is_continuous_vertical = True

                            # 2. MARK PADDING FOR VERTICAL TOWER (Strict 3x3)
                            l_start, l_end = min(prev_layer, layer), max(prev_layer, layer)
                            for l_idx in range(l_start, l_end + 1):
                                grid_3d[gx, l_idx, gz] = 1
                                for px in [-1, 0, 1]:
                                    for pz in [-1, 0, 1]:
                                        if 0 <= gx + px < width and 0 <= gz + pz < length:
                                            grid_3d[gx + px, l_idx, gz + pz] = 1

                            # 3. BUILD TOWER
                            low_y = min(LAYER_HEIGHTS[prev_layer], LAYER_HEIGHTS[layer])
                            high_y = max(LAYER_HEIGHTS[prev_layer], LAYER_HEIGHTS[layer])
                            is_ascending = layer > prev_layer

                            if is_ascending:
                                # Ascending: Base Wire -> Observers Down
                                range_limit = high_y + 1 if is_continuous_vertical else high_y
                                for h in range(low_y + 1, range_limit):
                                    schem_wrapper.setBlock((wx, h, wz), "minecraft:observer[facing=down]")

                                if not is_continuous_vertical:
                                    # Output Bulb at Top
                                    schem_wrapper.setBlock((wx, high_y, wz), "minecraft:copper_bulb")
                                    pending_comparator = True
                            else:
                                # Descending: Observers Up
                                range_limit = low_y - 1 if is_continuous_vertical else low_y
                                for h in range(high_y - 1, range_limit, -1):
                                    schem_wrapper.setBlock((wx, h, wz), "minecraft:observer[facing=up]")

                                if not is_continuous_vertical:
                                    # --- UPDATED DESCENDING LOGIC ---
                                    # 1. Output Bulb at Bottom (Destination)
                                    schem_wrapper.setBlock((wx, low_y, wz), "minecraft:copper_bulb")
                                    pending_comparator = True

                                    # 2. Input Bulb at Top (Entry)
                                    # Overwrite the wire that was placed at high_y
                                    schem_wrapper.setBlock((wx, high_y, wz), "minecraft:copper_bulb")

                                    # 3. Input Comparator
                                    # Backtrack to place comparator feeding INTO the top bulb
                                    if i >= 2:
                                        p_gx, p_layer, p_gz = path_indices[i - 2]
                                        # Ensure i-2 is on the same layer (horizontal approach)
                                        if p_layer == prev_layer:
                                            dx = gx - p_gx  # gx is current (Bulb), p_gx is previous (Comparator)
                                            dz = gz - p_gz

                                            c_facing = "north"
                                            if dx == 1:
                                                c_facing = "east"
                                            elif dx == -1:
                                                c_facing = "west"
                                            elif dz == 1:
                                                c_facing = "south"
                                            elif dz == -1:
                                                c_facing = "north"



                                            p_wx, p_wy, p_wz = p_gx + min_x, LAYER_HEIGHTS[p_layer], p_gz + min_z
                                            schem_wrapper.setBlock((p_wx, p_wy, p_wz),
                                                                   f"minecraft:comparator[facing={c_facing},mode=compare]")
                                    # --- NEW LOGIC END ---

                            signal_dist = 0
                            continue

                        # 2. Horizontal Wire (or Comparator)
                        block_type = "minecraft:redstone_wire"
                        signal_dist += 1

                        facing = "north"
                        if i < len(path_indices) - 1:
                            nx, _, nz = path_indices[i + 1]
                            if nz - gz == 1:
                                facing = "north"
                            elif nz - gz == -1:
                                facing = "south"
                            elif nx - gx == 1:
                                facing = "west"
                            elif nx - gx == -1:
                                facing = "east"
                        elif i > 0:
                            px, _, pz = path_indices[i - 1]
                            if gz - pz == 1:
                                facing = "north"
                            elif gz - pz == -1:
                                facing = "south"
                            elif gx - px == 1:
                                facing = "west"
                            elif gx - px == -1:
                                facing = "east"

                        if pending_comparator:
                            block_type = f"minecraft:comparator[facing={facing},mode=compare]"
                            pending_comparator = False
                            signal_dist = 0
                        elif signal_dist >= 15 and 0 < i < len(path_indices) - 1:
                            pgx, _, pgz = path_indices[i - 1]
                            ngx, _, ngz = path_indices[i + 1]
                            if not ((gx - pgx != ngx - gx) or (gz - pgz != ngz - gz)):
                                signal_dist = 0
                                r_facing = "north"
                                if facing == "south":
                                    r_facing = "north"
                                elif facing == "north":
                                    r_facing = "south"
                                elif facing == "east":
                                    r_facing = "west"
                                elif facing == "west":
                                    r_facing = "east"
                                block_type = f"minecraft:repeater[facing={r_facing}]"

                        schem_wrapper.setBlock((wx, wy - 1, wz), wool_block)
                        schem_wrapper.setBlock((wx, wy, wz), block_type)

                        if i == 0 and layer == 0 and usage_count == 0:
                            schem_wrapper.setBlock((wx, wy - 1, wz), wool_block)
                            schem_wrapper.setBlock((wx, wy, wz), "minecraft:redstone_wire")
                else:
                    print(f"Failed to route signal from {node.type} to {target_node.type} (3D routing failed)")
class AStarRouter:
    def __init__(self):
        pass

    def calculate_heuristic(self, pos: Tuple[int, int, int], start: Tuple[int, int, int],
                            end: Tuple[int, int, int]) -> float:
        # Reduced penalty from 20 to 5 to encourage layer usage
        dx = abs(pos[0] - end[0])
        dy = abs(pos[1] - end[1])
        dz = abs(pos[2] - end[2])
        return dx + (dy * 5) + dz

    def get_valid_neighbors(self, grid: np.ndarray, position: Tuple[int, int, int],
                            blocked_points: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        x, y, z = position
        width, layers, length = grid.shape
        possible_moves = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y, z + 1), (x, y, z - 1),
            (x, y + 1, z), (x, y - 1, z)
        ]
        results = []
        for nx, ny, nz in possible_moves:
            if 0 <= nx < width and 0 <= ny < layers and 0 <= nz < length:
                if grid[nx, ny, nz] == 0 and (nx, ny, nz) not in blocked_points:
                    results.append((nx, ny, nz))
        return results

    def reconstruct_path(self, goal_node: Dict) -> List[Tuple[int, int, int]]:
        path = []
        current = goal_node
        while current is not None:
            path.append(current['position'])
            current = current['parent']
        return path[::-1]

    def find_path(self, grid: np.ndarray,
                  start: Tuple[int, int, int], end: Tuple[int, int, int],
                  start_socket_def: Optional[Tuple[int, int, int]] = None,
                  end_socket_def: Optional[Tuple[int, int, int]] = None) -> List[Tuple[int, int, int]]:

        width, layers, length = grid.shape
        if not (0 <= start[0] < width and 0 <= start[1] < layers and 0 <= start[2] < length): return []
        if not (0 <= end[0] < width and 0 <= end[1] < layers and 0 <= end[2] < length): return []

        # --- NORMALS ---
        start_normal = None
        if start_socket_def:
            snx, snz = start_socket_def[0], start_socket_def[2]
            if snx != 0 or snz != 0: start_normal = (snx, 0, snz)

        end_normal = None
        if end_socket_def:
            enx, enz = end_socket_def[0], end_socket_def[2]
            if enx != 0 or enz != 0: end_normal = (enx, 0, enz)

        # --- CALCULATE TUNNELS ---
        s1, s2 = None, None
        if start_normal:
            s1 = (start[0] + start_normal[0], start[1], start[2] + start_normal[2])
            s2 = (start[0] + 2 * start_normal[0], start[1], start[2] + 2 * start_normal[2])

        e1, e2 = None, None
        if end_normal:
            e1 = (end[0] + end_normal[0], end[1], end[2] + end_normal[2])
            e2 = (end[0] + 2 * end_normal[0], end[1], end[2] + 2 * end_normal[2])

        # --- A* LOOP ---
        h_start = self.calculate_heuristic(start, start, end)
        start_node = {'position': start, 'g': 0, 'h': h_start, 'f': h_start, 'parent': None}

        open_list = [(start_node['f'], start)]
        open_dict = {start: start_node}
        closed_set = set()

        # Helper to check if a specific spot is valid
        def is_walkable(pt):
            px, py, pz = pt
            if 0 <= px < width and 0 <= py < layers and 0 <= pz < length:
                return grid[px, py, pz] == 0
            return False

        while open_list:
            _, current_pos = heapq.heappop(open_list)
            if current_pos in closed_set: continue

            if current_pos == end:
                # Optional: Valid Entry Check (relaxed for robustness)
                return self.reconstruct_path(open_dict[current_pos])

            closed_set.add(current_pos)
            current_node = open_dict[current_pos]

            # --- SMART NEIGHBOR SELECTION ---
            neighbors = []
            forced_move = None

            # 1. Start Tunnel Logic (Smart Fallback)
            if current_pos == start and s1:
                # Only force s1 if it is actually walkable!
                if is_walkable(s1):
                    forced_move = s1

            elif s1 and current_pos == s1 and s2:
                # Only force s2 if it is walkable!
                if is_walkable(s2):
                    forced_move = s2

            if forced_move:
                neighbors = [forced_move]
            else:
                # If forced move is blocked (or we are past the tunnel), use standard neighbors.
                # This allows the "Vertical Escape" (UP neighbor) to be found.
                neighbors = self.get_valid_neighbors(grid, current_pos, set())

            # --- EXPANSION ---
            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set: continue

                move_cost = 1
                # Vertical Penalty
                if neighbor_pos[1] != current_pos[1]: move_cost = 20
                # Turn Penalty
                if current_node['parent']:
                    px, py, pz = current_node['parent']['position']
                    if (neighbor_pos[0] != current_pos[0] and current_pos[0] == px) or \
                            (neighbor_pos[2] != current_pos[2] and current_pos[2] == pz):
                        move_cost += 2

                tentative_g = current_node['g'] + move_cost
                h_cost = self.calculate_heuristic(neighbor_pos, start, end)

                if neighbor_pos not in open_dict or tentative_g < open_dict[neighbor_pos]['g']:
                    neighbor_node = {
                        'position': neighbor_pos,
                        'g': tentative_g,
                        'h': h_cost,
                        'f': tentative_g + h_cost,
                        'parent': current_node
                    }
                    open_dict[neighbor_pos] = neighbor_node
                    heapq.heappush(open_list, (neighbor_node['f'], neighbor_pos))

        return []
if __name__ == "__main__":
    import importlib
    from yosys_parser import parse_yosys_json
    from Optimizer import Optimizer

    factory = RedstoneFactory()

    nodes = parse_yosys_json("counter.json", factory)
    optimizer = Optimizer(factory, bounds=(60, 60), startT=30000, endT=0.01, cooling_rate=0.998)
    optimizer.run(steps=50000)
    schematic = AmuletSchematic()
    factory.compile_to_schematic(schematic)
    schematic.save("devschematics", "redeploy")