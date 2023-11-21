from model import Model

input = [
    37.2,
    59.5,
    99.5,
    31.5,
    2505,
    118.23,
    188.88,
    7952.38,
    39.4,
    62.2,
    104.1,
    32.4,
    2675,
    118.23,
    188.88,
    7952.38,
    36.4,
    60.7,
    106.5,
    33.4,
    2828,
    121.73,
    191.97,
    8256.17,
    38.0,
    63.5,
    109.7,
    34.8,
    3037,
    108.93,
    181.73,
    8467.06,
    40.3,
    70.4,
    110.8,
    36.7,
    3239,
    109.22,
    182.47,
    8727.01,
    38.9,
    72.5,
    113.5,
    38.8,
    3489,
    109.81,
    191.82,
    8825.61,
    40.5,
    75,
    112.7,
    40.5,
    3740,
    100.42,
    186.85,
    8992.26,
    41.8,
    82.1,
    115.5,
    41.8,
    4000,
    123.84,
    231.53,
    10092.34,
    54.9,
    102.8,
    108.8,
    44.4,
    4481,
    107.91,
    207.09,
    9847.87,
    53.2,
    102.1,
    115.7,
    49.3,
    4855,
    110.33,
    191.07,
    9834.57,
    59.4,
    102.8,
    118.8,
    53.8,
    5291,
    100.02,
    185.18,
    9234.56,
    59.7,
    96.9,
    127.5,
    56.9,
    5744,
    100.14,
    196.41,
    9569.37,
    60.1,
    92,
    124,
    60.6,
    6262,
    104.92,
    170.29,
    10094.90,
    66.5,
    124.0,
    117.9,
    65.2,
    6968,
    99.17,
    151.81,
    10333.33,
    68.0,
    154.0,
    105.5,
    72.6,
    7682,
    101.99,
    190.21,
    10687.11,
    72.0,
    162.1,
    103.4,
    82.4,
    8421,
    93.66,
    212.12,
    10581.26,
    73.7,
    162.8,
    104.3,
    90.9,
    9243,
    87.37,
    196.81,
    10219.66,
    71.6,
    165.6,
    103.7,
    96.5,
    9724,
    81.07,
    179.16,
    10168.31,
    72.8,
    162.2,
    105.7,
    99.6,
    10340,
    74.19,
    171.59,
    10076.68,
    81.4,
    163.5,
    105.5,
    103.9,
    11257,
    73.09,
    162.82,
    10381.52,
    76.0,
    158.8,
    106.5,
    107.6,
    11861,
    78.34,
    157.40,
    10834.45,
    84.0,
    157.4,
    107.3,
    109.6,
    12469,
    70.63,
    147.55,
    11023.23,
    78.0,
    164.9,
    103.3,
    113.6,
    13094,
    76.64,
    143.62,
    11376.82,
    85.4,
    250.3,
    102.8,
    118.3,
    14477,
    68.66,
    145.16,
    11526.40,
    92.7,
    265.7,
    97.7,
    124,
    15307,
    72.16,
    211.61,
    12237.53,
    89.9,
    281.0,
    95.8,
    130.7,
    16205,
    74.75,
    214.24,
    12344.35,
    88.0,
    288.3,
    94.9,
    136.2,
    16766,
    68.79,
    215.00,
    12398.62,
    86.9,
    284.6,
    94.1,
    140.3,
    17636,
    64.63,
    211.69,
    12309.83,
    89.0,
    293.4,
    92.0,
    144.5,
    18153,
    61.95,
    202.85,
    12570.20,
    90.1,
    282.9,
    95.2,
    148.2,
    19003,
    61.60,
    203.07,
    12562.62,
    91.7,
    284.3,
    95.5,
    152.4,
    20604,
    60.78,
    190.87,
    12822.53,
    97.3,
    280.2,
    95.7,
    156.9,
    21375,
    60.15,
    186.57,
    13519.63,
    100.2,
    279.5,
    93.6,
    160.5,
    22312,
    61.99,
    178.60,
    13623.32,
    104.2,
    277.1,
    95.0,
    163,
    23016,
    62.42,
    174.16,
    13901.55,
    105.6,
    287.8,
    96.2,
    166.6,
    23693,
    64.03,
    170.01,
    14120.24,
    107.1,
    306.4,
    96.5,
    172.2,
    24908,
    63.37,
    172.72,
    14221.48,
]


def solve(i, j, k):
    additive = Model(input, "test.txt", 36, (2, 1, 2, 3), (i, j, k), "Чебишова", False)
    additive.solve()
    return additive.error_normalized


def find_best():
    errors = {}

    for i in range(8):
        for j in range(8):
            for k in range(8):
                errors.update({f"{i},{j},{k}": solve(i, j, k)})
                # errors |= {f"{i},{j},{k}": solve(i, j, k)}

    return min(errors, key=lambda k: errors[k].mean())


print(len(input))
print(find_best())
print(solve(6, 5, 1))
