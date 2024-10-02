import schemas
# Importer les fonctions des schémas depuis le fichier schemas.py

t1 = 2.5
t2 = 4.5

# Dimension du domaine
L = 10

# La vitesse de transport
a = 2

# Le coéfficient CFL
CFL = 0.8

"""
Cette fonction prend le nombre N de points de maillage,
le numero n = {1, 2, 3, 4} du schema et le temps final t de simulation
"""

def schema_numerique(N,t,n):
    if n==1:
        """Schéma centré"""
        schemas.centre(t, L, N, a, CFL)
        
    elif n==2:
        """Schéma decentré"""
        schemas.decentre(t, L, N, a, CFL)
        
    elif n==3:
        """Schéma de LaxFriedrichs"""
        schemas.LaxFriedrichs(t, L, N, a, CFL)
        
    elif n==4:
        """Schéma de LaxWendroff"""
        schemas.LaxWendroff(t, L, N, a, CFL)

# Nombre de noeuds
N = 200
schema_numerique(N,t2,4)

