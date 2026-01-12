import json
import random


def parse_yosys_json(json_path, factory):
    """
    Parses a Yosys JSON file, populates the Factory with components,
    and establishes all graph connections.

    Returns:
        List[ComponentInstance]: The list of created nodes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Get the top module (assuming 'counter' or the only module defined)
    modules = data.get("modules", {})
    if not modules:
        print("Error: No modules found in JSON.")
        return []

    # Grab the first module found
    module_name = list(modules.keys())[0]
    module = modules[module_name]
    print(f"Parsing module: {module_name}")

    # --- DATA STRUCTURES FOR WIRING ---
    # We need to map "Net IDs" (integers) to Nodes.
    # net_drivers[net_id] = (ComponentInstance, PinName)
    net_drivers = {}

    # net_receivers = [ (net_id, ComponentInstance, PinName) ]
    net_receivers = []

    # --- STEP A: HANDLE PORTS (External Inputs like Clk/Rst) ---
    ports = module.get("ports", {})

    # Just for visualization, we place generic IO blocks in a line
    io_cursor = 0

    for port_name, port_data in ports.items():
        direction = port_data["direction"]
        bits = port_data["bits"]  # List of Net IDs, e.g. [2]

        if direction == "input":
            # CHANGE "WIRE" TO "INPUT"
            input_node = factory.create("INPUT", -5, 4, io_cursor * 2)
            io_cursor += 1

            for bit_id in bits:
                net_drivers[bit_id] = (input_node, 'Y')
                print(f"  Port Input {port_name} (Net {bit_id}) -> Node {input_node.id}")

    # --- STEP B: CREATE CELLS (The Gates) ---
    cells = module.get("cells", {})

    # We'll scatter them randomly initially for the SA to fix later
    # Or place them in a naive grid
    grid_x = 0
    grid_z = 0
    GRID_STRIDE = 6

    for cell_name, cell_data in cells.items():
        cell_type = cell_data["type"]

        # 1. Map Yosys types to Our types
        # Yosys might name them "NOR", "$_NOR_", etc.
        my_type = cell_type.replace("$_", "").replace("_", "")  # Clean up names

        # Handle specific mappings if names don't match exactly
        if my_type == "DFFP": my_type = "DFF"  # Yosys DFF Positive edge

        # Create the node
        try:
            # Naive placement
            node = factory.create(my_type, grid_x, 4, grid_z)

            # Update grid for next one
            grid_x += GRID_STRIDE
            if grid_x > 30:
                grid_x = 0
                grid_z += GRID_STRIDE

        except ValueError:
            print(f"  Warning: Unknown cell type '{cell_type}'. Skipping.")
            continue

        # 2. Process Connections
        connections = cell_data["connections"]

        for pin_name, bits in connections.items():
            if not bits: continue
            net_id = bits[0]  # Usually single bit signals

            # Determine if this pin is an Input or Output based on Component Prototype
            # We look at the prototype to know direction
            if pin_name in node.proto.outputs:
                # This node DRIVES this net
                net_drivers[net_id] = (node, pin_name)
            elif pin_name in node.proto.inputs:
                # This node READS this net
                net_receivers.append((net_id, node, pin_name))
            else:
                # Fallback for Clock pins on DFF which might be labeled 'C' or 'CLK'
                # Check known aliases
                if my_type == "DFF" and pin_name in ["C", "CLK"]:
                    net_receivers.append((net_id, node, "C"))
                else:
                    print(f"  Warn: Unknown pin {pin_name} on {my_type}")

    # --- STEP C: CONNECT THE GRAPH ---
    print("Linking graph...")
    connection_count = 0

    for net_id, receiver_node, receiver_pin in net_receivers:
        # Who drives this wire?
        if net_id in net_drivers:
            driver_node, driver_pin = net_drivers[net_id]

            # Perform the connection in our graph
            driver_node.connect(receiver_node, driver_pin, receiver_pin)
            connection_count += 1
        else:
            # This happens for 'out' ports if we didn't create Output Nodes.
            # (Wires going to nowhere / world output)
            pass

    print(f"Successfully created {len(factory.graph.nodes)} components and {connection_count} connections.")
    return factory.graph.nodes


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Import your main redstone file
    # Assuming the file above is named 'redstone.py'
    from redstone import RedstoneFactory, AmuletSchematic

    # 1. Init
    factory = RedstoneFactory()

    # 2. Parse
    nodes = parse_yosys_json("counter.json", factory)

    # 3. Calculate Initial Energy (Probably terrible because random placement)
    print(f"Initial Energy: {factory.graph.calculate_total_energy()}")

    # 4. Save Initial mess to inspect
    schem = AmuletSchematic()
    factory.compile_to_schematic(schem)
    schem.save("devschematics", "parsed_initial_state")