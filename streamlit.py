import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import RobustScaler
import random

st.image('sdi.jpg')

# Configuración de Streamlit
st.title('Predicción de riesgo en Barcelona')

# Cargamos el modelo entrenado y el df
model_xgb = load('model_xgb.joblib')
df = pd.read_csv('dades_feb1.csv')

# Diccionario que mapea los números a los nombres de los espacios
nombres_espacios = {
    1: 'Plaça Catalunya',
    2: 'Ciutadella',
    3: 'Lluís Companys',
    4: 'Parc del Clot',
    5: 'Parc de la Guineueta',
    6: 'Jardins Blancafort',
    7: 'Plaça Lesseps',
    8: 'Jardins Laribal',
    9: 'Jardins Joan Maragall',
    10: 'Rambla Raval',
    11: 'Plaça Tiran lo Blanc',
    12: 'Parc del Poblenou',
    13: 'Parc de Sant Martí',
    14: 'Parc de la Pegaso',
    15: 'Parc Joan Miró',
    16: 'Espanya Industrial',
    17: 'Parc del Turó de la Peira',
    18: 'Parc del Carmel',
    19: 'Montjuïc',
    20: 'Parc Estació Nord',
    21: 'Plaça Tetuan',
    22: 'Plaça Espanya',
    23: 'Martí Codolar',
    24: 'Parc Putxet',
    25: 'Parc Sarrià',
    26: 'Plaça dels Àngels',
    27: 'Rambles',
    28: 'Carrer Hospital',
    29: 'Carrer Carretes',
    30: 'Carrer Cera',
    31: 'Parc Tres Xemeneies',
    32: 'Parc Florida',
    33: 'Plaça Catalunya',
    34: 'Carrer Bailen',
    35: 'Plaça universitat',
    36: 'Liceu',
    37: 'Plaça Vila de Gràcia',
    38: 'Plaça Joanic',
    39: 'Plaça Virreina',
    40: 'Plaça Heroïnes',
    41: 'Plaça Catalunya',
    42: 'Ciutadella',
    43: 'Lluís Companys',
    44: 'Parc del Clot',
    45: 'Parc de la Guineueta',
    46: 'Jardins Blancafort',
    47: 'Plaça Lesseps',
    48: 'Jardins Laribal',
    49: 'Jardins Joan Maragall',
    50: 'Rambla Raval',
    51: 'Plaça Tiran lo Blanc',
    52: 'Parc del Poblenou',
    53: 'Parc de Sant Martí',
    54: 'Parc de la Pegaso',
    55: 'Parc Joan Miró',
    56: 'Espanya Industrial',
    57: 'Parc del Turó de la Peira',
    58: 'Parc del Carmel',
    59: 'Montjuïc',
    50: 'Parc Estació Nord',
    51: 'Plaça Tetuan',
    52: 'Plaça Espanya',
    53: 'Martí Codolar',
    54: 'Parc Putxet',
    55: 'Parc Sarrià',
    56: 'Plaça dels Àngels',
    57: 'Rambles',
    58: 'Carrer Hospital',
    59: 'Carrer Carretes',
    60: 'Carrer Cera',
    61: 'Parc Tres Xemeneies',
    62: 'Parc Florida',
    63: 'Plaça de la Oca',
    64: 'Plaça Bonet i Muixí',
    65: 'Plaça Sants',
    66: 'Plaça Països catalans',
    67: 'Plaça Olzinelles',
    68: 'Rambla Brasil',
    69: 'Rambla Badal',
    70: 'Parc Estació Nord',
    71: 'Plaça Tetuan',
    72: 'Plaça Espanya',
    73: 'Martí Codolar',
    74: 'Parc Putxet',
    75: 'Parc Sarrià',
    76: 'Plaça dels Àngels',
    77: 'Rambles',
    78: 'Carrer Hospital',
    79: 'Carrer Carretes',
    80: 'Carrer Cera',
    81: 'Parc Tres Xemeneies',
    82: 'Parc Florida',
    83: 'Plaça Catalunya',
    84: 'Carrer Bailen',
    85: 'Plaça universitat',
    86: 'Liceu',
    87: 'Plaça Vila de Gràcia',
    88: 'Plaça Joanic',
    89: 'Plaça Virreina',
    90: 'Plaça Heroïnes',
    91: 'Plaça Catalunya',
    92: 'Ciutadella',
    93: 'Lluís Companys',
    94: 'Parc del Clot',
    95: 'Parc de la Guineueta',
    96: 'Jardins Blancafort',
    97: 'Plaça Lesseps',
    98: 'Jardins Laribal',
    99: 'Jardins Joan Maragall',
    90: 'Rambla Raval',
    91: 'Plaça Tiran lo Blanc',
    92: 'Parc del Poblenou',
    93: 'Parc de Sant Martí',
    94: 'Parc de la Pegaso',
    95: 'Parc Joan Miró',
    96: 'Espanya Industrial',
    97: 'Parc del Turó de la Peira',
    98: 'Parc del Carmel',
    99: 'Montjuïc',
    90: 'Parc Estació Nord',
    91: 'Plaça Tetuan',
    92: 'Plaça Espanya',
    93: 'Martí Codolar',
    94: 'Parc Putxet',
    95: 'Parc Sarrià',
    96: 'Plaça dels Àngels',
    97: 'Rambles',
    98: 'Carrer Hospital',
    99: 'Carrer Carretes',
    90: 'Carrer Cera',
    91: 'Parc Tres Xemeneies',
    92: 'Parc Florida',
    93: 'Plaça Bonet i Muixí',
    94: 'Rambla Badal',
    95: 'Plaça George Orwell',
    96: 'Plaça Oca',
    97: 'Plaça Diamant',
    98:'Carrer Verdi',
    99:'Carrer Sardà',
    100:'Carrer Foc',
    101: 'Plaça Catalunya',
    102: 'Ciutadella',
    103: 'Lluís Companys',
    104: 'Parc del Clot',
    105: 'Parc de la Guineueta',
    106: 'Jardins Blancafort',
    107: 'Plaça Lesseps',
    108: 'Jardins Laribal',
    109: 'Jardins Joan Maragall',
    110: 'Rambla Raval',
    111: 'Plaça Tiran lo Blanc',
    112: 'Parc del Poblenou',
    113: 'Parc de Sant Martí',
    114: 'Parc de la Pegaso',
    115: 'Parc Joan Miró',
    116: 'Espanya Industrial',
    117: 'Parc del Turó de la Peira',
    118: 'Parc del Carmel',
    119: 'Montjuïc',
    120: 'Parc Estació Nord',
    121: 'Plaça Tetuan',
    122: 'Plaça Espanya',
    123: 'Martí Codolar',
    124: 'Parc Putxet',
    125: 'Parc Sarrià',
    126: 'Plaça dels Àngels',
    127: 'Rambles',
    128: 'Carrer Hospital',
    129: 'Carrer Carretes',
    130: 'Carrer Cera',
    131: 'Parc Tres Xemeneies',
    132: 'Parc Florida',
    133: 'Plaça Catalunya',
    134: 'Carrer Bailen',
    135: 'Plaça universitat',
    136: 'Liceu',
    137: 'Plaça Vila de Gràcia',
    138: 'Plaça Joanic',
    139: 'Plaça Virreina',
    140: 'Plaça Heroïnes',
    141: 'Plaça Catalunya',
    142: 'Ciutadella',
    143: 'Lluís Companys',
    144: 'Parc del Clot',
    145: 'Parc de la Guineueta',
    146: 'Jardins Blancafort',
    147: 'Plaça Lesseps',
    148: 'Jardins Laribal',
    149: 'Jardins Joan Maragall',
    150: 'Rambla Raval',
    151: 'Plaça Tiran lo Blanc',
    152: 'Parc del Poblenou',
    153: 'Parc de Sant Martí',
    154: 'Parc de la Pegaso',
    155: 'Parc Joan Miró',
    156: 'Espanya Industrial',
    157: 'Parc del Turó de la Peira',
    158: 'Parc del Carmel',
    159: 'Montjuïc',
    150: 'Parc Estació Nord',
    151: 'Plaça Tetuan',
    152: 'Plaça Espanya',
    153: 'Martí Codolar',
    154: 'Parc Putxet',
    155: 'Parc Sarrià',
    156: 'Plaça dels Àngels',
    157: 'Rambles',
    158: 'Carrer Hospital',
    159: 'Carrer Carretes',
    160: 'Carrer Cera',
    161: 'Parc Tres Xemeneies',
    162: 'Parc Florida'
}

# Utilizar los nombres de espacios en el selectbox de Streamlit
espai_id = st.selectbox('Lugar:', list(nombres_espacios.values()))

# Invertir el diccionario para mapear los nombres de los espacios a números
espacios_numeros = {v: k for k, v in nombres_espacios.items()}

# Obtener el número correspondiente al nombre del lugar seleccionado
numero_seleccionado = espacios_numeros[espai_id]
homes = st.number_input('Número de hombres', min_value=0, max_value=500, step=1)
st.write('El número de chicos es ', homes)

dones = st.number_input('Número de mujeres', min_value=0, max_value=500, step=1)
st.write('El número de chicas es ', dones)

consum = st.slider('Consumo de drogas', min_value=0, max_value=6)

pernocta = st.toggle('Pernocta')
punt_trobada = st.toggle('Punto de Encuentro')
proximitat_recurs = st.toggle('Proximidad a recurso')

# Convertir booleanos a números
punt_trobada_numerico = 1 if punt_trobada else 0
pernocta_numerico = 1 if pernocta else 0
proximitat_recurs_numerico = 1 if proximitat_recurs else 0

if st.button('¿Hay riesgo?'):
    # Convertir los datos a un DataFrame
    data = {
        'espai_id': [numero_seleccionado],
        'punt_trobada': [punt_trobada_numerico],
        'proximitat_recurs': [proximitat_recurs_numerico],
        'lloc_pernocta': [pernocta_numerico],
        'homes': [homes],
        'dones': [dones],
        'consum': [consum]
    }
    input_data = pd.DataFrame(data)

    # Crear una instancia de RobustScaler
    scaler = RobustScaler()

    # Ajusta el escalador a los datos y transforma las columnas seleccionadas
    scaled_columns = ['homes', 'dones']
    input_data[scaled_columns] = scaler.fit_transform(input_data[scaled_columns])

    # Mostrar el resultado
    predict = model_xgb.predict(input_data)[0]
    if predict == 0: 
        st.markdown('No hay riesgo')
        st.image ('green.jpg')
    elif predict == 1:
        st.markdown('Riesgo medio')
        st.image('yellow.jpg')
    else:
        st.markdown('Riesgo alto')
        st.image('red.jpg')

