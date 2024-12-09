#!/usr/bin/env python
# coding: utf-8

# # Plot MET

# In[ ]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
from glob import glob

def plot_wind_data(nc_files, latitude_min, latitude_max, longitude_min, longitude_max, output_path):
    """
    Gera gráficos de componentes do vento para múltiplos arquivos NetCDF em intervalos de 6 horas.

    Parâmetros:
        - nc_files: lista de caminhos para arquivos NetCDF.
        - latitude_min, latitude_max: limites de latitude para o subset dos dados.
        - longitude_min, longitude_max: limites de longitude para o subset dos dados.
        - output_path: pasta onde os gráficos serão salvos.
    """
    os.makedirs(output_path, exist_ok=True)  # Garante que o diretório de saída existe

    for nc_file in nc_files:
        data = xr.open_dataset(nc_file)
        
        # Filtragem espacial
        data_subset = data.sel(latitude=slice(latitude_min, latitude_max), longitude=slice(longitude_min, longitude_max))
        lon = data_subset['longitude']
        lat = data_subset['latitude']
        
        # Definindo a escala de cor para a magnitude do vento
        norm = mcolors.Normalize(vmin=1, vmax=11)

        # Obtendo a data do arquivo para nomear os gráficos
        data_date = nc_file.split('/')[-1].split('.')[0]  # Ex: '20190401'

        # Intervalos de 6 horas (00:00, 06:00, 12:00, 18:00)
        for time_idx in [0, 6, 12, 18]:
            u_wind = data_subset['u10'].isel(time=time_idx)
            v_wind = data_subset['v10'].isel(time=time_idx)
            wind_magnitude = np.sqrt(u_wind**2 + v_wind**2)
            
            # Criando o gráfico
            plt.figure(figsize=(10, 6))
            quiver = plt.quiver(lon, lat, u_wind.squeeze(), v_wind.squeeze(), wind_magnitude.squeeze(), 
                                cmap='plasma', norm=norm, scale=50)

            # Coordenadas do ponto marcado
            x_lon = -39.959700
            x_lat = -22.430000
            plt.plot(x_lon, x_lat, 'kx', markersize=12, markeredgewidth=2, label='P-53') #Alterar para seu ponto

            # Adicionando títulos, labels e colorbar
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Componentes do vento {time_idx:02d}:00 {data_date[:4]}/{data_date[4:6]}/{data_date[6:]}')
            plt.colorbar(quiver, label='Velocidade do Vento (m/s)', extend='max')
            plt.legend()

            # Salvando a figura
            output_filename = f"{output_path}/{data_date}_{time_idx:02d}00.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {output_filename}")
            plt.close()

# Exemplo de uso
# Defina o caminho da pasta com os arquivos NetCDF e o caminho de saída das figuras
nc_folder = '{seu_caminho}/MEDSLIK_II_2.01/METOCE_INP/ORIGINAL/MET' #Alterar para seu caminho 
output_folder = ' {seu caminho_para_salvar_os_graficos/}' # Alterar para seu caminho
nc_files = sorted(glob(f'{nc_folder}/*.nc'))  # Lista todos os arquivos .nc na pasta

# Executa a função para gerar as figuras
plot_wind_data(
    nc_files=nc_files,
    latitude_min=-23,
    latitude_max=-21,
    longitude_min=-41,
    longitude_max=-39,
    output_path=output_folder
)


# # Plot OCE

# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import os
from glob import glob

def plot_ocean_currents(nc_files, latitude_min, latitude_max, longitude_min, longitude_max, output_path):
    """
    Gera gráficos das correntes oceânicas para múltiplos arquivos NetCDF com dados de superfície.

    Parâmetros:
        - nc_files: lista de caminhos para arquivos NetCDF.
        - latitude_min, latitude_max: limites de latitude para o subset dos dados.
        - longitude_min, longitude_max: limites de longitude para o subset dos dados.
        - output_path: pasta onde os gráficos serão salvos.
    """
    os.makedirs(output_path, exist_ok=True)  # Garante que o diretório de saída existe

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        
        # Extraindo a data do arquivo para nomear os gráficos
        ano = ds.time.dt.year.item()
        mes = ds.time.dt.month.item()
        dia = ds.time.dt.day.item()
        data_str = f"{dia:02d}/{mes:02d}/{ano}"

        # Subset espacial
        subset = ds.sel(longitude=slice(longitude_min, longitude_max), latitude=slice(latitude_min, latitude_max))

        # Seleciona componentes U e V na superfície para o subset
        U_surface = subset['uo'].isel(time=0, depth=0)
        V_surface = subset['vo'].isel(time=0, depth=0)

        # Calcula a magnitude da velocidade para o colormap
        speed = np.sqrt(U_surface**2 + V_surface**2)
        x = subset['longitude']
        y = subset['latitude']

        # Configuração do plot com Cartopy
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
        
        # Adiciona características geográficas
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='darkgray', zorder=1)
        ax.add_feature(cfeature.COASTLINE, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)

        # Define ticks e rótulos dos eixos
        ax.set_xticks(np.arange(longitude_min, longitude_max + 1, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(latitude_min, latitude_max + 1, 1), crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Criação do quiver plot para correntes oceânicas
        quiver = ax.quiver(x, y, U_surface, V_surface, speed, scale=10, cmap='viridis')

        # Coordenadas do ponto marcado
        x_coord = -39.959700  # Longitude do ponto
        y_coord = -22.430000  # Latitude do ponto
        ax.plot(x_coord, y_coord, 'kx', markersize=12, markeredgewidth=2, label='P-53')

        # Adiciona um colorbar e título do plot com a data
        plt.colorbar(quiver, label='Velocidade (m/s)')
        ax.set_title(f'Correntes Oceânicas na Superfície {data_str}')
        ax.legend()

        # Salva a figura usando a data no nome do arquivo
        nome_figura = f"{output_path}/correntes_{ano}{mes:02d}{dia:02d}.png"
        plt.savefig(nome_figura, dpi=300, bbox_inches='tight')
        print(f"Figura salva: {nome_figura}")
        
        # Fecha a figura para liberar memória
        plt.close(fig)

# Exemplo de uso
# Defina o caminho da pasta com os arquivos NetCDF e o caminho de saída das figuras
nc_folder = '/home/locoste/MEDSLIK_II_2.01/METOCE_INP/ORIGINAL/OCE'  # Pasta dos arquivos NetCDF
output_folder = '/seu_caminho_para_salvar_os_graficos/'  # Pasta para salvar os gráficos
nc_files = sorted(glob(f'{nc_folder}/mercatorpsy4v3r1_gl12_mean_*.nc'))  # Lista todos os arquivos .nc na pasta

# Executa a função para gerar as figuras
plot_ocean_currents(
    nc_files=nc_files,
    latitude_min=-23,
    latitude_max=-21,
    longitude_min=-42,
    longitude_max=-39,
    output_path=output_folder
)


# # Figura Balanço de massa

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

def converter_fte_para_csv_formatado(input_file, output_file):
    """
    Converte um arquivo .fte para .csv e ajusta os dados ao formato esperado.
    Args:
        input_file (str): Caminho do arquivo .fte de entrada.
        output_file (str): Caminho para salvar o arquivo .csv ajustado.
    """
    # Ler o arquivo completo
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Detectar o cabeçalho e corrigir espaços
    header = None
    start_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):  # Detecta onde a tabela começa
            header = line.split()
            header = [col.replace("vol spilt", "vol_spilt") for col in header]
            start_line = i + 1  # A tabela começa após esta linha
            print(f"Cabeçalho corrigido: {header}")
            break

    # Garantir que o cabeçalho foi detectado
    if header is None:
        raise ValueError("Cabeçalho da tabela não encontrado no arquivo.")

    # Contar colunas de cada linha e corrigir inconsistências
    fixed_lines = []
    for i, line in enumerate(lines[start_line:], start=start_line + 1):
        columns = line.split()
        if len(columns) < len(header):
            columns.extend([np.nan] * (len(header) - len(columns)))
        elif len(columns) > len(header):
            columns = columns[:len(header)]
        fixed_lines.append(columns)

    # Criar o DataFrame com as colunas do cabeçalho corrigido
    df = pd.DataFrame(fixed_lines, columns=header)

    # Converter colunas numéricas
    numeric_columns = ['%evap', '%srf', '%srftot', '%disp', '%cstfxd', '%csttot']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ajustar os valores para o formato esperado (escala de 0 a 1.000.000)
    for col in numeric_columns:
        df[col] = df[col] * 1_000_000  # Exemplo de escala ajustada, personalize conforme necessário

    # Salvar o DataFrame formatado como CSV
    df.to_csv(output_file, index=False)
    print(f"Tabela formatada salva com sucesso em: {output_file}")

def criar_grafico_balanco_massa(csv_path, output_path, title):
    """
    Cria um gráfico de balanço de massa a partir de um arquivo CSV ajustado.
    Args:
        csv_path (str): Caminho do arquivo CSV de entrada.
        output_path (str): Caminho para salvar a imagem do gráfico.
        title (str): Título do gráfico.
    """
    # Ler o arquivo CSV
    data = pd.read_csv(csv_path)

    # Criar o gráfico
    plt.figure(figsize=(10, 6))

    # Adicionar gráficos de linha para cada variável
    sns.lineplot(data=data, x='time', y='%evap', linewidth=3, label=f'% Evaporated')
    sns.lineplot(data=data, x='time', y='%srf', linewidth=3, label=f'% Surface')
    sns.lineplot(data=data, x='time', y='%disp', linewidth=3, label=f'% Dispersed in water')
    sns.lineplot(data=data, x='time', y='%cstfxd', linewidth=3, label=f'% Shoreline')

    # Ajustar limites do eixo Y dinamicamente
    y_max = data[['%evap', '%srf', '%disp', '%cstfxd']].max().max()
    ax = plt.gca()
    ax.set_ylim(0, y_max)
    ax.set_xlim(left=0)

    # Configuração do gráfico
    ax.yaxis.set_major_locator(ticker.LinearLocator(6))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x / 10_000)}"))
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(title, fontsize=16)
    plt.xlabel("Time (hours)", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)

    # Salvar o gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")

    plt.show()

# Configurações para os arquivos
input_fte = "/home/locoste/rotinas/medslik.fte"  # Arquivo .fte
output_csv = "tabela_convertida_formatada.csv"  # Arquivo .csv gerado
output_image = "grafico_balanco_massa_formatado.png"  # Caminho para salvar o gráfico
titulo_grafico = "Mass balance - Sylvestre- API 28 (Medium)"  # Título do gráfico

# Executar a conversão e o gráfico
converter_fte_para_csv_formatado(input_fte, output_csv)
criar_grafico_balanco_massa(output_csv, output_image, titulo_grafico)


# # Cria Shapefiles

# In[ ]:


import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def gerar_shapefile_por_tempo(netcdf_path, shapefile_output, dia=None, hora=None, crs="EPSG:4326"):
    """
    Gera um shapefile a partir de um arquivo NetCDF 'spill_properties', permitindo a seleção por dia e hora específicos.
    
    Args:
        netcdf_path (str): Caminho para o arquivo NetCDF.
        shapefile_output (str): Caminho para salvar o shapefile.
        dia (str, opcional): Dia no formato 'YYYY-MM-DD'. Se None, considera todos os tempos.
        hora (str, opcional): Hora no formato 'HH:MM:SS'. Se None, usa o início do dia.
        crs (str, opcional): Sistema de referência espacial. Padrão é WGS 84 (EPSG:4326).
    """
    # Abrir o arquivo NetCDF
    data = xr.open_dataset(netcdf_path)

    # Verificar se a coluna 'time' existe
    if "time" not in data:
        raise ValueError("A variável 'time' não foi encontrada no arquivo NetCDF.")

    # Construir o tempo selecionado
    if dia is not None:
        if hora is None:
            hora = "00:00:00"  # Padrão é o início do dia
        time_selection = f"{dia}T{hora}"
        print(f"Selecionando o tempo: {time_selection}")
        
        # Selecionar o tempo no formato ISO 8601
        try:
            data = data.sel(time=time_selection)
        except KeyError:
            raise ValueError(f"Tempo '{time_selection}' não encontrado no arquivo.")
    
    # Extrair latitudes e longitudes
    if 'latitude' not in data or 'longitude' not in data:
        raise ValueError("O arquivo NetCDF não contém as variáveis 'latitude' ou 'longitude'.")
    
    latitude = data['latitude'].values.flatten()
    longitude = data['longitude'].values.flatten()

    # Criar DataFrame com as coordenadas
    df = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude
    }).dropna()  # Remover valores NaN

    # Criar geometria com shapely
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

    # Converter para GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Definir sistema de referência espacial (CRS)
    gdf.set_crs(crs, inplace=True)

    # Salvar como shapefile
    gdf.to_file(shapefile_output)
    print(f"Shapefile criado com sucesso: {shapefile_output}")


# Exemplo de uso
netcdf_file = "spill_properties.nc"  # Caminho do arquivo NetCDF
shapefile_path = "spill_simulation_day_hour.shp"  # Caminho para salvar o shapefile
dia = "2024-01-05"  # Data desejada no formato YYYY-MM-DD
hora = "12:00:00"  # Horário desejado no formato HH:MM:SS

# Executar a rotina
gerar_shapefile_por_tempo(netcdf_file, shapefile_path, dia=dia, hora=hora)

