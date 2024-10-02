import numpy as np
import matplotlib.pyplot as plt


def graphe(x, uex, u, Tfinal, t, N):
    # Cette fonction pour faire les graphes
    
    plt.figure(figsize=(20,15))
    plt.ylim(-0.5, 1)
    plt.plot(x,uex,'-g', label=f'Solution exacte t = {Tfinal}')
    plt.plot(x,u, '-r', label=f'Solution numérique t = {round(t,2)}')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Comparaison entre la solution exacte et numérique (N={N})')
    plt.pause(0.1)
    plt.show()
   

def f(u):
    return 0.5*(u**2)
    
def u0(x):
    if x < 2:
        return 0.4
    return 0.1


def burgers(N, Tfinal, schema):
    """
    Cette fonction prend en paramètre le nombre N de points du maillage,
    le temps final de simulation et le numero du schema, schema = {1, 2, 3, 4}
    """
    L = 6
    CFL = 0.8

    dx = L/(N-1)
    dt = CFL*dx
    lamda = dt/dx

    x = np.linspace(0, L, N)
    uex = np.zeros(N)  # La solution exacte
    u = np.zeros(N)    # La solution initiale pour le schéma numérique
    unew = np.zeros(N)  # La solution numérique calculée à un instant n

    for i in range(N):
        
        """Calcul de la solution initiale à t = 0"""
        u[i] = u0(x[i])
        
        """Calcul de la solution exacte"""
        if (x[i]-2)/Tfinal < (0.5)/2:
            uex[i] = 0.4
            
        else:
            uex[i] = 0.1
        
        
    """
    Les schémas numériques
    
    """
    if schema==1:
        """Schéma 1 (centré)"""
        
        # Boucle principale en temps
        t = 0
        while t <= Tfinal:
            
            umax = max(abs(u))
            dt = CFL*dx/umax
            lamda = dt/dx
            
            for i in range(1, N-1):
                # Calcul de la solution numérique
                
                unew[i] = u[i] - 0.5*lamda*u[i]*( u[i+1] - u[i-1] )
                
            # Condition initiale
            unew[0] = unew[1]
            unew[N-1] = unew[N-2]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) # Plot
                
            t += dt
            
    elif schema==2:
        """Schéma décentré en amont"""
        
        # Boucle principale en temps
        t = 0
        while t <= Tfinal:
            
            umax = max(abs(u))
            dt = CFL*dx/umax
            lamda = dt/dx
            
            for i in range(1,N-1):
                unew[i] = u[i] - lamda*( f(u[i]) - f(u[i-1]) )

            # Conditions aux limites
            unew[N-1] = unew[N-2]
            unew[0] = unew[1]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) # Plot
                
            t += dt
            
    elif schema==3:
        """Schéma de Lax-Friedrichs"""
        
        # Boucle principale en temps
        t = 0
        while t <= Tfinal:
            
            umax = max(abs(u))
            dt = CFL*dx/umax
            lamda = dt/dx

            for i in range(1,N-1):
                unew[i] = 0.5*(u[i-1] + u[i+1]) - 0.5*lamda*( f(u[i+1]) - f(u[i-1]) )
                
            # Conditions aux limites
            unew[N-1] = unew[N-2]
            unew[0] = unew[1]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) # plot
            
            t+=dt
        
    elif schema==4:
        """Schéma de Lax-Wendroff"""
        
        # Boucle principale en temps
        t = 0
        while t < Tfinal:
            
            umax = max(abs(u))
            dt = CFL*dx/umax
            lamda = dt/dx
            
            for i in range(1,N-1):
                
                alpha = 1 - ((dt*(u[i+1] - u[i])) / (2*dx)) 
                unew[i] = u[i] +  u[i]*(lamda**2/2)*(f(u[i-1]) - 2*f(u[i]) + f(u[i+1])) - alpha*(lamda/2)*(f(u[i+1]) - f(u[i-1]))
                

            # Conditions aux limites
            unew[N-1] = unew[N-2]
            unew[0] = unew[1]
            u = unew.copy()
            
            if t <= Tfinal:
                erreur = np.linalg.norm(uex-u, ord = 1) # Calcul de l'erreur en norme 1
                graphe(x, uex, u, Tfinal, t, N) # Plot
                
            t+=dt
            
    print("L'erreur d'approximation est : ", erreur)
        
        
t1 = 2.5
t2 = 4.5
N = 200
#N =200
burgers(N, t2, 1)


