import numpy as np

from src.grid import create_grid
from src.derivatives import derivative_x, derivative_y, laplacian

N = 20

x, y, X, Y, h = create_grid(N)

# Função de teste
f = X**2 + Y**2

# Derivadas numéricas
df_dx = derivative_x(f, h)
df_dy = derivative_y(f, h)
lap = laplacian(f, h)

# Derivadas exatas nos pontos internos
exact_dx = 2 * X[1:-1, 1:-1]
exact_dy = 2 * Y[1:-1, 1:-1]
exact_lap = 4 * np.ones_like(lap)

# Erros máximos
print("\nDerivada numérica em x:")
print(df_dx)

print("\nDerivada exata em x:")
print(exact_dx)

print("\nDerivada numérica em y:")
print(df_dy)

print("\nDerivada exata em y:")
print(exact_dy)
print("Erro em df/dx:", np.max(np.abs(df_dx - exact_dx)))
print("Erro em df/dy:", np.max(np.abs(df_dy - exact_dy)))
print("Erro no Laplaciano:", np.max(np.abs(lap - exact_lap)))
