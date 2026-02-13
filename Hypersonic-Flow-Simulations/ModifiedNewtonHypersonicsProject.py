import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt 

def parse_lawgs_file(filename):
    """
    Parses a LaWGS file to extract geometry components into 3D NumPy arrays.
    """
    geometry_data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("'") and line.endswith("'"):
            component_name = line.strip("'")
            i += 1
            header = lines[i].strip().split()
            nline = int(header[1])
            npnt = int(header[2])
            total_coords = nline * npnt * 3
            points_list = []
            i += 1
            while len(points_list) < total_coords and i < len(lines):
                try:
                    points_list.extend([float(val) for val in lines[i].strip().split()])
                except ValueError:
                    pass
                i += 1
            points_array = np.array(points_list)
            reshaped_array = points_array.reshape(nline, npnt, 3)
            geometry_data[component_name] = reshaped_array
            i -= 1
        i += 1
    return geometry_data

def calculate_panel_properties(geometry_dict):
    """
    Calculates the centroid, area, and unit normal for each panel of the geometry.
    """
    panel_properties = {}
    for name, component_data in geometry_dict.items():
        n_lines, n_points, _ = component_data.shape
        num_panels_i = n_lines - 1
        num_panels_j = n_points - 1
        centroids, areas, normals = [], [], []
        for i in range(num_panels_i):
            for j in range(num_panels_j):
                p1 = component_data[i, j, :]
                p2 = component_data[i+1, j, :]
                p3 = component_data[i+1, j+1, :]
                p4 = component_data[i, j+1, :]
                centroid = (p1 + p2 + p3 + p4) / 4.0
                centroids.append(centroid)
                vec_diag1 = p3 - p1
                vec_diag2 = p4 - p2
                cross_product = np.cross(vec_diag1, vec_diag2)
                norm_magnitude = np.linalg.norm(cross_product)
                panel_area = 0.5 * norm_magnitude
                areas.append(panel_area)
                if norm_magnitude > 1e-9:
                    unit_normal = cross_product / norm_magnitude
                else:
                    unit_normal = np.array([0., 0., 0.])
                normals.append(unit_normal)
        panel_properties[name] = {
            'centroids': np.array(centroids),
            'areas': np.array(areas),
            'normals': np.array(normals)
        }
    return panel_properties

def calculate_dot_products(panel_data, alpha_deg):
    """
    Calculates the dot product of a freestream velocity vector with each panel normal.
    """
    alpha_rad = np.radians(alpha_deg)
    v_hat = np.array([np.cos(alpha_rad), 0, np.sin(alpha_rad)])
    dot_product_results = {}
    for name, properties in panel_data.items():
        normals = properties['normals']
        dot_products = np.dot(normals, v_hat)
        dot_product_results[name] = dot_products
    return dot_product_results

def calculate_cp_max(mach, gamma=1.4):
    """
    Calculates the max pressure coefficient behind a normal shock using the Rayleigh-Pitot formula.
    """
    m2 = mach**2
    g = gamma
    term1_num = ((g + 1)**2 * m2)
    term1_den = (4 * g * m2 - 2 * (g - 1))
    term1 = (term1_num / term1_den)**(g / (g - 1))
    term2 = (1 - g + 2 * g * m2) / (g + 1)
    cp_max = (2 / (g * m2)) * (term1 * term2 - 1)
    return cp_max

def calculate_pressure_coefficients(dot_product_dict, cp_max):
    """
    Calculates the pressure coefficient on each panel using Modified Newtonian Theory.
    """
    cp_results = {}
    for name, dot_products in dot_product_dict.items():
        effective_dot_products = np.where(dot_products < 0, dot_products, 0)
        cp_values = cp_max * (effective_dot_products**2)
        cp_results[name] = cp_values
    return cp_results

def calculate_total_forces_and_coeffs(panel_data, cp_data, alpha_deg):
    """
    Calculates the total force coefficient vector and resolves it into Lift and Drag coefficients.
    """
    total_force_coeff_vector = np.array([0.0, 0.0, 0.0])
    for name in panel_data:
        areas = panel_data[name]['areas']
        normals = panel_data[name]['normals']
        cp_values = cp_data[name]
        force_coeff_vectors = -cp_values[:, np.newaxis] * normals * areas[:, np.newaxis]
        total_force_coeff_vector += np.sum(force_coeff_vectors, axis=0)
    alpha_rad = np.radians(alpha_deg)
    v_hat = np.array([np.cos(alpha_rad), 0, np.sin(alpha_rad)])/Sref
    l_hat = np.array([-np.sin(alpha_rad), 0, np.cos(alpha_rad)])/Sref
    c_d = np.dot(total_force_coeff_vector, v_hat)
    c_l = np.dot(total_force_coeff_vector, l_hat)
    return total_force_coeff_vector, c_l, c_d

def plot_interactive_panels_plotly(geometry_dict, panel_data_dict, dot_product_dict, cp_data_dict, alpha_deg):
    """
    Creates an interactive 3D plot with hover data including dot product and Cp.
    """
    fig = go.Figure()
    all_centroids_x, all_centroids_y, all_centroids_z, all_hover_texts = [], [], [], []
    wire_x, wire_y, wire_z = [], [], []

    for component_data in geometry_dict.values():
        n_lines, n_points, _ = component_data.shape
        for line_idx in range(n_lines):
            wire_x.extend(component_data[line_idx, :, 0]); wire_x.append(None)
            wire_y.extend(component_data[line_idx, :, 1]); wire_y.append(None)
            wire_z.extend(component_data[line_idx, :, 2]); wire_z.append(None)
        for pt_idx in range(n_points):
            wire_x.extend(component_data[:, pt_idx, 0]); wire_x.append(None)
            wire_y.extend(component_data[:, pt_idx, 1]); wire_y.append(None)
            wire_z.extend(component_data[:, pt_idx, 2]); wire_z.append(None)
    fig.add_trace(go.Scatter3d(x=wire_x, y=wire_y, z=wire_z, mode='lines', line=dict(color='black', width=2), name='Wireframe', hoverinfo='none'))

    for name, component_data in geometry_dict.items():
        n_lines, n_points, _ = component_data.shape
        panel_props, dot_products, cp_values = panel_data_dict[name], dot_product_dict[name], cp_data_dict[name]
        all_centroids_x.extend(panel_props['centroids'][:, 0])
        all_centroids_y.extend(panel_props['centroids'][:, 1])
        all_centroids_z.extend(panel_props['centroids'][:, 2])
        panel_idx = 0
        for i in range(n_lines - 1):
            for j in range(n_points - 1):
                p1, p2, p3, p4 = component_data[i, j, :], component_data[i+1, j, :], component_data[i+1, j+1, :], component_data[i, j+1, :]
                c, a, n, dp_value, cp_value = panel_props['centroids'][panel_idx], panel_props['areas'][panel_idx], panel_props['normals'][panel_idx], dot_products[panel_idx], cp_values[panel_idx]
                text = (f"<b>Panel {name}-{panel_idx}</b><br>"
                        f"------------------<br>"
                        f"<b>Pressure Coeff (Cp): {cp_value:.4f}</b><br>"
                        f"<b>Dot Product (α={alpha_deg}°): {dp_value:.4f}</b><br>"
                        f"Centroid: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})<br>"
                        f"Area: {a:.4f}<br>"
                        f"Normal: ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})")
                all_hover_texts.append(text)
                panel_idx += 1
    fig.add_trace(go.Scatter3d(x=all_centroids_x, y=all_centroids_y, z=all_centroids_z, mode='markers', hoverinfo='text', text=all_hover_texts, marker=dict(color='red', size=2, opacity=0.7), name='Panel Centroids'))
    fig.update_layout(title=f'Interactive 3D Geometry | Hover for Panel Data | α = {alpha_deg}°', scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Z-axis', aspectmode='data'), legend=dict(x=0, y=1, traceorder='normal'), margin=dict(r=10, l=10, b=10, t=50))
    fig.show(renderer="browser")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. DEFINE CONSTANT CONDITIONS & GEOMETRY ---
    MACH_NUMBER = 4.63
    GAMMA = 1.4
    wgs_file = 'tmx1242.wgs'
    Sref = 100
    # --- 2. PERFORM ONE-TIME CALCULATIONS ---
    print("Parsing geometry and calculating panel properties...")
    aircraft_geometry = parse_lawgs_file(wgs_file)
    panel_data = calculate_panel_properties(aircraft_geometry)
    cp_max = calculate_cp_max(MACH_NUMBER, GAMMA)
    print(f"One-time calculations complete. Cp_max = {cp_max:.4f}")

    # --- 3. GENERATE INTERACTIVE 3D PLOT (PLOTLY) ---
    VISUALIZATION_ALPHA = 0.0
    print(f"\nGenerating interactive 3D model for α = {VISUALIZATION_ALPHA}°...")
    vis_dot_products = calculate_dot_products(panel_data, VISUALIZATION_ALPHA)
    vis_cp_data = calculate_pressure_coefficients(vis_dot_products, cp_max)
    plot_interactive_panels_plotly(aircraft_geometry, panel_data, vis_dot_products, vis_cp_data, VISUALIZATION_ALPHA)

    # --- 4. RUN ANGLE OF ATTACK SWEEP FOR 2D PLOT ---
    alpha_array_deg = np.arange(0, 41, 1) 
    cl_results = []
    cd_results = []

    print(f"\nStarting angle of attack sweep from {alpha_array_deg[0]} to {alpha_array_deg[-1]} degrees...")
    for alpha_deg in alpha_array_deg:
        dot_products = calculate_dot_products(panel_data, alpha_deg)
        cp_data = calculate_pressure_coefficients(dot_products, cp_max)
        _, c_l, c_d = calculate_total_forces_and_coeffs(panel_data, cp_data, alpha_deg)
        cl_results.append(c_l)
        cd_results.append(c_d)
        print(f"  Calculated for Alpha = {alpha_deg}° -> CL = {c_l:.4f}, CD = {c_d:.4f}")

    print("\nAngle of attack sweep complete.")

    # --- 5. PLOT CL & CD vs ALPHA (MATPLOTLIB) ---
    print("Generating final 2D plot using Matplotlib...")
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_array_deg, cl_results, 'o-', label='$C_L$ (Lift Coefficient)')
    plt.plot(alpha_array_deg, cd_results, 's-', label='$C_D$ (Drag Coefficient)')
    plt.title(f'Aerodynamic Coefficients vs. Angle of Attack (Mach = {MACH_NUMBER})')
    plt.xlabel('Angle of Attack, α (degrees)')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.legend()
    plt.show()