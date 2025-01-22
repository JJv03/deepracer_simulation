import csv
import xml.etree.ElementTree as ET

def extract_waypoints_to_csv(dae_file, output_csv, array_id='road-verts-array', step=10):
    """
    Extrae waypoints desde un archivo .dae y los guarda en un archivo CSV.

    :param dae_file: Ruta al archivo .dae
    :param output_csv: Ruta al archivo de salida .csv
    :param array_id: ID del array que contiene las coordenadas (por defecto, 'road-verts-array')
    :param step: Intervalo para seleccionar waypoints (por defecto, 10)
    """
    # Cargar y analizar el archivo .dae
    tree = ET.parse(dae_file)
    root = tree.getroot()

    # Buscar el array de coordenadas con el ID proporcionado
    waypoints = []
    for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
        if float_array.attrib.get('id') == array_id:
            data = list(map(float, float_array.text.strip().split()))

            # Las coordenadas vienen en grupos de tres (x, y, z)
            for i in range(0, len(data), 3 * step):
                waypoint = data[i:i+3]  # Extraer x, y, z
                if len(waypoint) == 3:
                    waypoints.append(waypoint)
            break

    # Guardar los waypoints en un archivo CSV
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'z'])  # Encabezados
        writer.writerows(waypoints)  # Escribir las filas de waypoints

    print(f"Waypoints extraídos y guardados en {output_csv}")

# Parámetros de entrada
dae_file = "../meshes/2022_april_open/2022_april_open.dae"  # Archivo proporcionado
output_csv = "waypoints.csv"  # Archivo de salida
array_id = "cl-verts-array"  # "cl-verts-array" (linea central) - "road-verts-array" (Carretera) - "ol-verts-array" (Linea exterior)
step = 1  # Intervalo para seleccionar waypoints

# Ejecutar la extracción (valores organizados grupos de tres, 5940/3=1980 puntos)
extract_waypoints_to_csv(dae_file, output_csv, array_id, step)
