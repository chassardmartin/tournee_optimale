# Attrape le voleur !

Un voleur se déplace dans les rues d'une ville. On ne le voit pas directement : on ne reçoit que des
**indices** imprécis sur sa position. Le **super-héros** utilise ces indices pour deviner où est le voleur
et le rattraper (c'est un *filtre de Kalman étendu*).

## Lancer le jeu

```bash
pip install numpy scipy matplotlib
python3 ekf_video.py
```

## Comment jouer

1. Le voleur 🔴 apparaît sur un carrefour.
2. **Clique sur un carrefour** pour placer le super-héros 🔵 au départ.
3. La poursuite commence : le voleur laisse des indices 🟡, et le super-héros s'en sert pour le suivre.
   La zone bleue claire montre où il pense que se trouve le voleur.
4. Fin de la partie :
   - **« Le voleur a été rattrapé ! »** : le super-héros est arrivé tout près du voleur ;
   - **« Le voleur s'est échappé ! »** : il ne l'a pas retrouvé à temps.
5. **Clique sur la carte** : un nouveau voleur apparaît, et tu peux rejouer.

Pendant une poursuite, un clic sur un autre carrefour la recommence depuis ce carrefour.

## Niveaux de difficulté

```bash
python3 ekf_video.py easy     # facile : le voleur est presque toujours rattrapé
python3 ekf_video.py hard     # indices plus imprécis
python3 ekf_video.py          # « tricky » (par défaut) : super-héros têtu, rattrapage plus lent
```

Avec `--save`, une vidéo de la première partie est enregistrée en mp4 au lieu de jouer (nécessite `ffmpeg`).
