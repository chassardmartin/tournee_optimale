# Attrape le voleur !

Un voleur se déplace dans les rues d'une ville. On ne le voit pas directement : on ne reçoit que des
**indices** imprécis sur sa position. La **super-héroïne** utilise ces indices pour deviner où est le voleur
et le rattraper (c'est un *filtre de Kalman étendu*).

## Lancer le jeu

```bash
pip install numpy scipy matplotlib
python3 ekf_video.py
```

## Comment jouer

1. Le voleur 🥐 (un pain au chocolat, entouré de rouge) apparaît sur un carrefour.
2. **Clique sur un carrefour** pour placer la super-héroïne 🔵 (entourée de bleu) au départ.
3. La poursuite commence : le voleur laisse des indices 🟡, et la super-héroïne s'en sert pour le suivre.
   Les traits rouge et bleu montrent leurs chemins.
4. Fin de la partie :
   - **« Le voleur a été rattrapé ! »** : la super-héroïne est arrivée tout près du voleur ;
   - **« Le voleur s'est échappé ! »** : elle ne l'a pas retrouvé à temps.
5. **Clique sur la carte** : un nouveau voleur apparaît, avec un nouveau chemin, et tu peux rejouer.

Pendant une poursuite, un clic sur un autre carrefour la recommence depuis ce carrefour.

## Niveaux de difficulté

```bash
python3 ekf_video.py          # « tricky » (par défaut) : super-héroïne têtue, environ 4 voleurs sur 5 rattrapés
python3 ekf_video.py easy     # facile : le voleur est presque toujours rattrapé
python3 ekf_video.py hard     # indices plus imprécis
```

Avec `--save`, une vidéo de la première partie est enregistrée en mp4 au lieu de jouer (nécessite `ffmpeg`).

## Images

Les images sont dans `images/` : `pain_au_chocolat.jpg` (voleur), `super_hero.png` (super-héroïne),
`house.png` et `tree.png` (la ville). Pour les changer, modifier `CHARACTERS` et `CITY_IMAGES` au début de
`ekf_video.py`.
