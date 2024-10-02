import numpy as np
import matplotlib.pyplot as plt

"""
Ce fichier contient les fonctions qui réalisent les différents schémas.
Chaque fonction prend en paramètre le temps final de simulation,
la longueur L du domaine, le nombre N de points du maillage,
la vitesse de transport et le coefficient CFL.

Puisque on ne peut pas représenter la solution numérique à
t = 2.5 ou t = 4.5 exactement, nous représentons la solution
numérique à un temps très proche de de ces temps physiques.

"""


def u0(x): # Solution initiale
    if 3 <= x <= 4:
        return 1
    return 0

def graphe(x, uex, u, Tfinal, t, N):
    # Cette fonction pour faire les graphes
    
    plt.figure(figsize=(6,4))
    plt.ylim(-0.5, 2)
    plt.plot(x,uex,'-g', label=f'Solution exacte t = {Tfinal}')
    plt.plot(x,u, '-r', label=f'Solution numérique t = {round(t,2)}')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Comparaison entre la solution exacte et numérique (N={N})')
    plt.pause(0.001)
    plt.show()
    


def centre(Tfinal, L, N, a, CFL):
    # Cette fonction réalise le schéma centré

    x = np.linspace(0,L,N)
    uex = np.zeros(N) # La solution exacte
    u = np.zeros(N) # La solution numérique
    unew = np.zeros(N)
    
    for i in range(N):
        u[i] = u0(x[i])
        
        # Calculer la solution exacte au temps final de simulation
        b = x[i] - a*Tfinal
        
        if 0 <= b <= L:
            uex[i] = u0(b)
            
        elif b > L:
            b = b - L
            uex[i] = u0(b)
            
        else:
            b = b + L
            uex[i] = u0(b)
        
    dx = L/(N-1)
    dt = CFL*dx/abs(a)
    lamda = a*dt/dx
    
    # Boucle principale en temps
    t = 0
    while t < Tfinal:
        for i in range(1,N-1):
            unew[i] = u[i] - 0.5*lamda*( u[i+1] - u[i-1] )

        # Conditions aux limites
        unew[N-1] = u[N-1]
        unew[0] = unew[N-1]
        u = unew.copy()
        
        if t <= Tfinal:
            erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
            graphe(x, uex, u, Tfinal, t, N) # Plot
            
        t+=dt
        
    print("L'erreur d'approximation est : ", erreur)
    
        
        
def decentre(Tfinal, L, N, a, CFL):
    # Cette fonction réalise le schéma decentré
    
    x = np.linspace(0,L,N)
    uex = np.zeros(N) # La solution exacte
    u = np.zeros(N) # La solution numérique
    unew = np.zeros(N)
    
    for i in range(N):
        u[i] = u0(x[i])
        
        # Calculer la solution exacte au temps final de simulation
        b = x[i] - a*Tfinal
        if 0 <= b <= L:
            uex[i] = u0(b)
            
        elif b > L:
            b = b - L
            uex[i] = u0(b)
            
        else:
            b = b + L
            uex[i] = u0(b)
        
    dx = L/(N-1)
    dt = CFL*dx/abs(a)
    lamda = a*dt/dx
    
    if a > 0:
        # Boucle principale en temps
        t = 0
        while t < Tfinal:
            for i in range(1,N-1):
                unew[i] = u[i] - lamda*( u[i] - u[i-1] )

            # Conditions aux limites
            unew[N-1] = unew[N-2]
            unew[0] = unew[N-1]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) # Plot
                
            t += dt
            
        print("L'erreur d'approximation est : ", erreur)
        
        

    if a < 0:
        # Boucle principale en temps
        t = 0
        while t < Tfinal:
            
            for i in range(1,N-1):
                unew[i] = u[i] - lamda*( u[i+1] - u[i] )

            # Conditions aux limites
            unew[N-1] = unew[N-2]
            unew[0] = unew[N-1]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) #Plot
                
            t += dt
            
        print("L'erreur d'approximation est : ", erreur)
            


def LaxFriedrichs(Tfinal, L, N, a, CFL):
    # Cette fonction réalise le schéma de LaxFriedrichs
    
    dx = L/(N-1)
    dt = CFL*dx/abs(a)
    lamda = a*dt/dx
    
    uex = np.zeros(N) # Solution exacte
    u = np.zeros(N) # Solution numérique
    unew = np.zeros(N)
    x = np.linspace(0,L,N)

    for i in range(N):
        u[i] = u0(x[i])
        
        # Calculer la solution exacte au temps final de simulation
        b = x[i] - a*Tfinal
        if 0 <= b <= L:
            uex[i] = u0(b)
            
        elif b > L:
            b = b - L
            uex[i] = u0(b)
            
        else:
            b = b + L
            uex[i] = u0(b)
       
    # Boucle principale en temps
    t = 0
    while t < Tfinal:

        for i in range(1,N-1):
            unew[i] = 0.5*(u[i-1] + u[i+1]) - 0.5*lamda*( u[i+1] - u[i-1] )

        # Conditions aux limites
        unew[N-1] = unew[N-2]
        unew[0] = unew[N-1]
        u = unew.copy()
        
        if t <= Tfinal:
            erreur = np.linalg.norm(uex-u, ord = 1)  # Calcul de l'erreur en norme 1
            graphe(x, uex, u, Tfinal, t, N) # Plot
        
        t+=dt
        
    print("L'erreur d'approximation est : ", erreur)
    


def LaxWendroff(Tfinal, L, N, a, CFL):
    # Cette fonction réalise le schéma de LaxWendroff
    
    dx = L/(N-1)
    dt = CFL*dx/abs(a)
    lamda = a*dt/dx
    
    x = np.linspace(0,L,N)
    uex = np.zeros(N) # La solution exacte
    u = np.zeros(N) # La solution numérique
    unew = np.zeros(N)

    for i in range(N):
        u[i] = u0(x[i])
        
        # Calculer la solution exacte au temps final de simulation
        b = x[i] - a*Tfinal
        if 0 <= b <= L:
            uex[i] = u0(b)
            
        elif b > L:
            b = b - L
            uex[i] = u0(b)
            
        else:
            b = b + L
            uex[i] = u0(b)
        
    # Boucle principale en temps
    t = 0
    while t < Tfinal:
        for i in range(1,N-1):
            unew[i] = u[i] - 0.5*lamda*(u[i+1] - u[i-1]) + 0.5*(lamda**2)*( u[i-1] - 2*u[i] + u[i+1] )

        # Conditions aux limites
        unew[N-1] = unew[N-2]
        unew[0] = unew[N-1]
        u = unew.copy()
        
        if t <= Tfinal:
            erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
            graphe(x, uex, u, Tfinal, t, N) # Plot
            
        t+=dt
        
    print("L'erreur d'approximation est : ", erreur)

