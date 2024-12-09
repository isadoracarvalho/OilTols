#!/usr/bin/env python
# coding: utf-8

# # Baixar dados OCE

# In[ ]:


import requests
from datetime import datetime, timedelta

# Parâmetros do usuário
username = 'seu_usuario'  # Substitua pelo seu usuário
password = 'sua_senha'    # Substitua pela sua senha
output_path = '{seu_caminho}/MEDSLIK_II_2.01/METOCE_INP/ORIGINAL/OCE/'  # Especifique o caminho de saída
url_base = 'https://nrt.cmems-du.eu/motu-web/Motu'  # URL base do Copernicus Marine Service

# Intervalo de datas para download (5 dias como exemplo)
data_inicio = datetime(2023, 11, 1)  # Ajuste a data de início
numero_de_dias = 5  # Número de dias que deseja baixar

# Função para download
def baixar_dados_diarios(data_inicio, numero_de_dias):
    data_atual = data_inicio
    for _ in range(numero_de_dias):
        data_str = data_atual.strftime('%Y%m%d')  # Formata a data para o padrão desejado (YYYYMMDD)
        
        # Nome do arquivo personalizado
        output_filename = f'{output_path}mercatorpsy4v3r1_gl12_mean_{data_str}.nc'
        
        params = {
            'service-id': 'GLOBAL_MULTIYEAR_PHY_001_030-TDS',
            'product-id': 'cmems_mod_glo_phy_my_0.083_P1D-m',
            'date-min': data_atual.strftime('%Y-%m-%d'),
            'date-max': data_atual.strftime('%Y-%m-%d'),
            'variable': ['uo', 'vo', 'thetao'],  # Componentes u, v das correntes e temperatura
            'longitude-min': -180, 'longitude-max': 180,  # Ajuste conforme sua área de interesse
            'latitude-min': -90, 'latitude-max': 90,      # Ajuste conforme sua área de interesse
            'depth-min': 0, 'depth-max': 5727.91796875,   # Profundidade ajustável
            'motu-user': username,
            'motu-password': password,
            'output-filename': output_filename,
        }
        
        response = requests.get(url_base, params=params, auth=(username, password))
        
        if response.status_code == 200:
            with open(output_filename, 'wb') as file:
                file.write(response.content)
            print(f"Arquivo {output_filename} baixado com sucesso!")
        else:
            print(f"Erro ao baixar {output_filename}: {response.status_code}")
        
        # Avança para o próximo dia
        data_atual += timedelta(days=1)

# Executa o download para os dias especificados
baixar_dados_diarios(data_inicio, numero_de_dias)


# # Baixar dados MET

# ## Pré-requisitos
# 
# Conta no Climate Data Store (CDS): Certifique-se de estar registrado no Copernicus Climate Data Store e tenha uma chave API configurada no seu computador.
# 

# ### Biblioteca cdsapi: Instale a biblioteca cdsapi com o seguinte comando:

# In[ ]:


pip install cdsapi


# ### Configuração da API: Coloque seu arquivo de configuração .cdsapirc no diretório inicial com suas credenciais. O arquivo .cdsapirc deve ter o seguinte formato:

# In[ ]:


url: https://cds.climate.copernicus.eu/api/v2
key: sua_chave_de_api


# In[ ]:


import cdsapi
from datetime import datetime, timedelta

# Inicializa o cliente da API do CDS
c = cdsapi.Client()

# Parâmetros do usuário
output_path = '{seu_caminho}/MEDSLIK_II_2.01/METOCE_INP/ORIGINAL/MET'  # Especifique o caminho de saída
data_inicio = datetime(2023, 11, 1)  # Data de início
numero_de_dias = 5  # Número de dias para baixar

# Área de interesse [latitude_norte, longitude_oeste, latitude_sul, longitude_leste]
area = [5, -60, -35, -20]  # Alterar o grid

# Função para download
def baixar_dados_era5(data_inicio, numero_de_dias):
    data_atual = data_inicio
    for _ in range(numero_de_dias):
        data_str = data_atual.strftime('%Y%m%d')  # Formato de data AAAAMMDD
        
        output_filename = f'{output_path}{data_str}.nc'  # Salva apenas com a data AAAAMMDD
        
        # Requisição para baixar dados diários de vento (u10 e v10) para a área definida
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': data_atual.strftime('%Y'),
                'month': data_atual.strftime('%m'),
                'day': data_atual.strftime('%d'),
                'time': [
                    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
                    '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', 
                    '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
                ],
                'area': area  # Define a área de interesse
            },
            output_filename
        )
        
        print(f"Arquivo {output_filename} baixado com sucesso!")
        
        # Avança para o próximo dia
        data_atual += timedelta(days=1)

# Executa o download para os dias especificados
baixar_dados_era5(data_inicio, numero_de_dias)

