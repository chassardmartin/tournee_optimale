import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse, Circle
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from PIL import Image
from scipy import ndimage

### my own imports
from grid import Grid2D
from particle_dynamics import GridParticle
from kalman_filters import EKF
from ekf_particle import h, dh, observe

### functions
def covariance_ellipse(mean, cov, n_std=2.0):
    """
    parameters of the n_std-sigma ellipse of a 2D covariance
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(eigvals[::-1])
    return mean, width, height, angle

### colour code: the thief is red, the superhero (EKF) blue, the clues (observations) yellow,
### the catch green, the city (grid) soft gray
COLORS = dict(
    background='#f7f7f2',
    street='#d5d8dc',
    node='#aab7c4',
    thief='#e74c3c',
    hero='#2e86de',
    clue='#f1c40f',
    catch='#27ae60',
    escape='#e74c3c',
)

### pictures of the characters (any format readable by Pillow), drawn `size` grid units wide in place of their dot.
### The thief is cut out of its background and keeps a red halo; the superhero is a round badge (crop `circle` =
### (centre x, centre y, radius) as fractions of the picture height) with a blue ring.
### Without the file, the character is a plain dot
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
CHARACTERS = {
    'true_dot': dict(image=os.path.join(IMAGES_DIR, 'pain_au_chocolat.jpg'), size=0.8),
    'est_dot': dict(image=os.path.join(IMAGES_DIR, 'super_hero.png'), size=0.65, circle=(0.44, 0.44, 0.31)),
}

def load_sprite(path, size=256, circle=None):
    """
    load an image as a square RGBA array of size x size pixels, cropped to its content.
    With circle = (cx, cy, r), the picture is cropped to that disc (fractions of the picture height for r).
    Otherwise an image without transparency (e.g. a jpg photo) gets its light, colourless background removed:
    the light gray pixels connected to the border of the image become transparent
    """
    rgba = np.asarray(Image.open(path).convert('RGBA')).astype(float) / 255
    alpha = rgba[..., 3]
    if circle is not None:
        h, w = alpha.shape
        cx, cy, r = circle[0] * w, circle[1] * h, circle[2] * h
        yy, xx = np.mgrid[:h, :w]
        # antialiased disc: alpha goes from 1 to 0 over the last pixel of the radius
        alpha = alpha * np.clip(r - np.hypot(xx + 0.5 - cx, yy + 0.5 - cy), 0, 1)
    elif alpha.min() == 1:
        rgb = rgba[..., :3]
        value, saturation = rgb.max(2), np.ptp(rgb, axis=2) / np.maximum(rgb.max(2), 1e-6)
        labels, _ = ndimage.label((saturation < 0.08) & (value > 0.55))
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        background = np.isin(labels, border[border > 0])
        # shave the pale fringe of the shadow, then soften the edge
        alpha = ndimage.gaussian_filter(ndimage.binary_erosion(~background, iterations=2).astype(float), 1.5)
    rows, cols = np.nonzero(alpha > 0.05)
    rgba = np.dstack([rgba[..., :3], alpha])[rows.min():rows.max()+1, cols.min():cols.max()+1]

    # pad to a square, centred, then downscale
    h, w = rgba.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 4))
    square[(side - h)//2:(side - h)//2 + h, (side - w)//2:(side - w)//2 + w] = rgba
    square = Image.fromarray((square * 255).astype(np.uint8), 'RGBA').resize((size, size), Image.LANCZOS)
    return np.asarray(square)

class HandlerImage(HandlerBase):
    """
    legend entry drawn as a picture
    """
    def __init__(self, image, scale=2.2):
        super().__init__()
        self.image = image
        self.scale = scale

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        size = self.scale * height
        bbox = Bbox.from_bounds(-xdescent + (width - size) / 2, -ydescent + (height - size) / 2, size, size)
        image = BboxImage(TransformedBbox(bbox, trans))
        image.set_data(self.image)
        return [image]

def move_character(artists, key, x, y):
    """
    draw the character key ('true_dot' for the thief, 'est_dot' for the superhero) at (x, y):
    its dot, and its picture and ring if it has some
    """
    artists[key].set_data([x], [y])
    if key + '_img' in artists:
        r = CHARACTERS[key]['size'] / 2
        artists[key + '_img'].set_extent((x - r, x + r, y - r, y + r))
        artists[key + '_img'].set_visible(True)
    if key + '_ring' in artists:
        artists[key + '_ring'].set_center((x, y))
        artists[key + '_ring'].set_visible(True)

### the city: some blocks between the streets get a house, a tree or two small trees, drawn from these pictures
### (png versions of the svg files of images/, made with
###   convert -background none -density 384 images/house.svg -resize 256x256 images/house.png)
### The layout is random but fixed by CITY_SEED, so the city is the same in every game
CITY_IMAGES = dict(house=os.path.join(IMAGES_DIR, 'house.png'), tree=os.path.join(IMAGES_DIR, 'tree.png'))
CITY_SEED = 3
# fraction of the blocks with a house, a tree, two trees (the others stay empty)
CITY_DENSITY = dict(house=0.20, tree=0.10, two_trees=0.05)

def draw_city(ax, grid, px_per_unit=80):
    """
    draw the houses and trees in the blocks of the grid. They are pasted once into a single picture covering the
    map, so that the animation does not redraw 100 small images at every frame
    """
    if not all(os.path.exists(path) for path in CITY_IMAGES.values()):
        return
    rng = np.random.default_rng(CITY_SEED)
    pictures = {name: Image.fromarray(load_sprite(path, size=128)) for name, path in CITY_IMAGES.items()}
    x0, y1 = grid.x_min - 0.5, grid.y_max + 0.5 # top-left corner of the picture
    width = int(round((grid.xlength + 1) * px_per_unit))
    height = int(round((grid.ylength + 1) * px_per_unit))
    city = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    def paste(name, x, y, size):
        # centre (x, y) and size in grid units
        side = int(round(size * px_per_unit))
        picture = pictures[name].resize((side, side), Image.LANCZOS)
        left = int(round((x - x0) * px_per_unit)) - side // 2
        top = int(round((y1 - y) * px_per_unit)) - side // 2
        city.alpha_composite(picture, (left, top))

    for i in range(grid.Nx):
        for j in range(grid.Ny):
            # centre of the block between nodes (i, j) and (i+1, j+1)
            x = grid.x_min + (i + 0.5) * grid.dx
            y = grid.y_min + (j + 0.5) * grid.dy
            u = rng.random()
            if u < CITY_DENSITY['house']:
                paste('house', x, y, 0.55)
            elif u < CITY_DENSITY['house'] + CITY_DENSITY['tree']:
                paste('tree', x, y, 0.55)
            elif u < sum(CITY_DENSITY.values()):
                paste('tree', x - 0.17, y + 0.08, 0.38)
                paste('tree', x + 0.17, y - 0.08, 0.38)

    # above the streets (zorder 1), below the crossroads (zorder 2)
    ax.imshow(np.asarray(city), extent=(x0, grid.x_max + 0.5, grid.y_min - 0.5, y1), zorder=1.5,
              interpolation='antialiased')

def setup_plot(ax, grid):
    """
    draw the grid and create the (empty) artists of the animation
    """
    c = COLORS
    ax.figure.set_facecolor(c['background'])
    ax.set_facecolor(c['background'])
    grid.plot_grid(ax=ax, node_size=90, node_color=c['node'], line_color=c['street'], line_width=3)
    draw_city(ax, grid)
    # pictures of the characters (created before setting the limits, imshow changes them)
    sprites, images = {}, {}
    for key, zorder in [('true_dot', 7.5), ('est_dot', 8.5)]:
        ch = CHARACTERS[key]
        if os.path.exists(ch['image']):
            sprites[key] = load_sprite(ch['image'], circle=ch.get('circle'))
            images[key] = ax.imshow(sprites[key], extent=(0, 1, 0, 1), zorder=zorder, interpolation='antialiased')
            images[key].set_visible(False)
    ax.set_xlim(grid.x_min - 0.5, grid.x_max + 0.5)
    ax.set_ylim(grid.y_min - 0.5, grid.y_max + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    artists = dict(
        obs_past=ax.scatter([], [], s=25, color=c['clue'], edgecolor='#b7950b', lw=0.5, alpha=0.7, zorder=3,
                            label='indices'),
        true_line=ax.plot([], [], color=c['thief'], lw=3, zorder=4, label='chemin du voleur')[0],
        # with a picture, the red dot becomes a halo under it, to keep the colour code
        true_dot=ax.plot([], [], 'o', color=c['thief'], mec='white', mew=2,
                         ms=28 if 'true_dot' in sprites else 18, alpha=0.35 if 'true_dot' in sprites else 1,
                         zorder=7, label='voleur')[0],
        est_line=ax.plot([], [], color=c['hero'], lw=3, zorder=5, label='chemin de la super-héroïne')[0],
        est_dot=ax.plot([], [], 'o', color=c['hero'], mec='white', mew=2, ms=18, zorder=8, label='super-héroïne')[0],
        prior_dot=ax.plot([], [], 'o', mfc='none', mec=c['hero'], mew=3, ms=22, zorder=6,
                          label='départ de la super-héroïne')[0],
        ellipse=ax.add_patch(Ellipse((0, 0), 0, 0, facecolor=c['hero'], edgecolor=c['hero'], alpha=0.15,
                                     zorder=4, label='zone de recherche')),
        title=ax.set_title('', fontsize=16, fontweight='bold', color='#34495e'),
        banner=ax.text(0.5, 0.5, '', transform=ax.transAxes, ha='center', va='center', fontsize=30,
                       fontweight='bold', color='white', zorder=10,
                       bbox=dict(boxstyle='round,pad=0.6', facecolor=c['catch'], edgecolor='white', lw=3)),
    )
    artists['banner'].set_visible(False)
    for key, image in images.items():
        artists[key + '_img'] = image
    if 'est_dot' in sprites:
        # the superhero badge replaces the blue dot, and gets a blue ring to keep the colour code
        artists['est_dot'].set_visible(False)
        artists['est_dot_ring'] = ax.add_patch(Circle((0, 0), CHARACTERS['est_dot']['size'] / 2, fill=False,
                                                      edgecolor=c['hero'], lw=3, zorder=8.6, visible=False))
    # one legend column per character: thief, superhero, start, clues
    order = ['true_dot', 'true_line', 'est_dot', 'est_line', 'prior_dot', 'ellipse', 'obs_past']
    handler_map = {artists[key]: HandlerImage(sprite) for key, sprite in sprites.items()}
    ax.legend([artists[k] for k in order], [artists[k].get_label() for k in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=10,
              markerscale=0.7, columnspacing=1.0, handler_map=handler_map)
    return artists

CATCH_MESSAGE = "Le voleur a été rattrapé !"
ESCAPE_MESSAGE = "Le voleur s'est échappé !"

def make_update(artists, states, observations, ekf, tau, catch_tol=1e-1):
    """
    update function of the animation.
    Frame 0 shows the prior, frame k >= 1 the estimate after the k-th correction.
    The thief is caught at the first frame where the position error is below catch_tol: the animation
    stops there. Returns the update function, the last frame to play and whether the thief is caught.
    """
    estimates = np.array([ekf.states_minus[0]] + ekf.states_plus)
    covs = np.array([ekf.Pi_minus_list[0]] + ekf.Pi_plus_list)[:, :2, :2]
    true_idx = np.maximum(np.arange(len(estimates)) - 1, 0) # time index of the true state for each frame
    errors = np.linalg.norm(estimates[:, :2] - states[true_idx, :2], axis=1)
    caught = bool((errors[1:] < catch_tol).any()) # the prior itself does not count
    last_frame = 1 + int(np.argmax(errors[1:] < catch_tol)) if caught else len(estimates) - 1

    a = artists
    a['prior_dot'].set_data([estimates[0, 0]], [estimates[0, 1]])
    a['banner'].set_visible(False)

    def update(frame):
        k = frame
        n = max(k - 1, 0) # time index of the true state
        a['obs_past'].set_offsets(observations[:k].reshape(-1, 2))
        a['true_line'].set_data(states[:n+1, 0], states[:n+1, 1])
        move_character(a, 'true_dot', states[n, 0], states[n, 1])
        a['est_line'].set_data(estimates[:k+1, 0], estimates[:k+1, 1])
        move_character(a, 'est_dot', estimates[k, 0], estimates[k, 1])

        center, width, height, angle = covariance_ellipse(estimates[k, :2], covs[k])
        a['ellipse'].set_center(center)
        a['ellipse'].set_width(width)
        a['ellipse'].set_height(height)
        a['ellipse'].set_angle(angle)

        label = 'départ' if k == 0 else f'temps : {n * tau:4.1f} s'
        a['title'].set_text(f'{label}    distance au voleur : {errors[k]:.2f}')
        if k == last_frame:
            # banner in the half of the map away from the super-hero, so it does not hide the end of the chase
            ymin, ymax = a['banner'].axes.get_ylim()
            a['banner'].set_y(0.2 if estimates[k, 1] > (ymin + ymax) / 2 else 0.8)
            a['banner'].set_text(CATCH_MESSAGE if caught else ESCAPE_MESSAGE)
            a['banner'].get_bbox_patch().set_facecolor(COLORS['catch'] if caught else COLORS['escape'])
            a['banner'].set_visible(True)
        return list(a.values())

    return update, last_frame, caught

def frame_sequence(last_frame, fps, hold_seconds):
    """
    hold the prior for 1s and the last frame for hold_seconds
    """
    return [0] * fps + list(range(last_frame + 1)) + [last_frame] * (hold_seconds * fps)

def make_video(grid, states, observations, ekf, tau, filename, fps=10, hold_seconds=2):
    """
    save the animation of the catch to an mp4 file
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    artists = setup_plot(ax, grid)
    update, last_frame, caught = make_update(artists, states, observations, ekf, tau)
    anim = FuncAnimation(fig, update, frames=frame_sequence(last_frame, fps, hold_seconds), blit=False)
    anim.save(filename, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
    print(f"video saved to {filename}")
    print(CATCH_MESSAGE if caught else ESCAPE_MESSAGE)

def clear_plot(artists):
    """
    remove everything drawn by a previous game (the grid stays)
    """
    artists['obs_past'].set_offsets(np.empty((0, 2)))
    for k in ['true_line', 'true_dot', 'est_line', 'est_dot', 'prior_dot']:
        artists[k].set_data([], [])
    artists['ellipse'].set_width(0)
    artists['ellipse'].set_height(0)
    artists['banner'].set_visible(False)
    for key in ['true_dot_img', 'est_dot_img', 'est_dot_ring']:
        if key in artists:
            artists[key].set_visible(False)

def interactive(grid, new_game, run_ekf, tau, fps=10):
    """
    single window. A game: the thief appears on its starting node, the user clicks on a node (snapped to
    the nearest one) to choose the prior position, the EKF is run from there and the catch is played live,
    frozen as soon as the thief is caught. Clicking during the chase restarts it from the new node, clicking
    once it is over starts a new game with a new thief.
    """
    _xx, _yy = grid.vals
    nodes = np.column_stack([_xx.ravel(), _yy.ravel()])
    fig, ax = plt.subplots(figsize=(8, 8))
    artists = setup_plot(ax, grid)
    # a single animation ticks for the whole session, a click only swaps the run it plays
    run = dict(game=None, update=None, frames=[], i=0, caught=False)

    def start_game():
        game = new_game()
        run.update(game=game, update=None, frames=[], i=0, caught=False)
        clear_plot(artists)
        move_character(artists, 'true_dot', game['states'][0, 0], game['states'][0, 1])
        artists['title'].set_text('Clique sur un carrefour pour placer la super-héroïne')
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if run['update'] is not None and run['i'] >= len(run['frames']): # game over
            start_game()
            return
        game = run['game']
        node = nodes[np.argmin(np.linalg.norm(nodes - [event.xdata, event.ydata], axis=1))]
        update, last_frame, caught = make_update(artists, game['states'], game['observations'],
                                                 run_ekf(game, *node), tau)
        run.update(update=update, frames=frame_sequence(last_frame, fps, hold_seconds=0), i=0, caught=caught)

    def step(_):
        if run['update'] is None or run['i'] >= len(run['frames']):
            return []
        out = run['update'](run['frames'][run['i']])
        run['i'] += 1
        if run['i'] == len(run['frames']):
            print(CATCH_MESSAGE if run['caught'] else ESCAPE_MESSAGE)
            artists['title'].set_text(artists['title'].get_text() + '\nClique sur la carte pour rejouer')
        return out

    start_game()
    fig.canvas.mpl_connect('button_press_event', on_click)
    anim = FuncAnimation(fig, step, frames=itertools.count(), interval=1000 / fps,
                         cache_frame_data=False, blit=False)
    plt.show()

def random_route(grid, rng, min_segment=2, max_segment=5, length_range=(14, 20)):
    """
    random self-avoiding route along the grid lines: straight segments of min_segment to max_segment edges
    joined by left/right turns. No node is visited twice, so each turning node is crossed only once.
    Returns (turns, start_node, theta0, stop_node), or None if the random walk got stuck.
    """
    L, R = np.pi/2, -np.pi/2
    directions = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)} # heading = quarter turns from east
    node = (int(rng.integers(0, grid.Nx + 1)), int(rng.integers(0, grid.Ny + 1)))
    heading = int(rng.integers(4))
    start, theta0 = node, heading * np.pi/2
    target = int(rng.integers(length_range[0], length_range[1] + 1))
    visited, turns, length = {node}, {}, 0

    while length < target:
        if length > 0: # turn at the end of the previous segment
            delta = L if rng.random() < 0.5 else R
            turns[node] = delta
            heading = (heading + (1 if delta == L else -1)) % 4
        n_edges = min(int(rng.integers(min_segment, max_segment + 1)), target - length)
        if n_edges < min_segment:
            break
        di, dj = directions[heading]
        for _ in range(n_edges):
            node = (node[0] + di, node[1] + dj)
            if not (0 <= node[0] <= grid.Nx and 0 <= node[1] <= grid.Ny) or node in visited:
                return None
            visited.add(node)
        length += n_edges
    if length < length_range[0]:
        return None
    turns.pop(node, None) # the walk may have turned at its last node before stopping: no turn there
    return turns, start, theta0, node


### scenarios: 'hard' trusts the observations less (bigger Q) and the model more (smaller W).
### 'tricky' (the default) keeps the observations of 'easy' but makes the filter stubborn: a very confident
### prior (small Pi0) and a trusted model (small position noise in W), so it is slow to let go of a wrong prior.
### Over random games and priors (catch margin 0.1): easy catches ~99% of thieves, median at 4 s;
### hard ~67%, median 5.6 s; tricky ~79%, median 8.3 s (about half-way through the thief's route)
SCENARIOS = {
    'easy': dict(sigma_obs=0.20, Pi0=0.1 * np.eye(3), W=np.diag([1e-2, 1e-2, 0.3]), filename='ekf_catch.mp4'),
    'hard': dict(sigma_obs=0.25, Pi0=0.1 * np.eye(3), W=np.diag([5e-3, 5e-3, 0.3]), filename='ekf_catch_hard.mp4'),
    'tricky': dict(sigma_obs=0.20, Pi0=1e-3 * np.eye(3), W=np.diag([1e-3, 1e-3, 0.3]),
                   filename='ekf_catch_tricky.mp4'),
}

### route of the first game (and of the saved video): no loop, from (1, 1) heading east, stops on node (6, 8)
L, R = np.pi/2, -np.pi/2
FIRST_ROUTE = ({(5, 1): L, (5, 4): R, (8, 4): L, (8, 8): L}, (1, 1), 0.0, (6, 8))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != '--save']
    save = '--save' in sys.argv[1:]
    scenario = SCENARIOS[args[0] if args else 'tricky']
    rng = np.random.default_rng(0)
    grid = Grid2D(x_min=0, x_max=10, y_min=0, y_max=10, Nx=10, Ny=10)

    sigma_obs = scenario['sigma_obs']        # std of one position measurement
    tau = 0.1
    Q = sigma_obs**2 * tau * np.eye(2)       # observation noise, scaled by 1/tau in the EKF
    W = scenario['W']                        # model noise, scaled by tau in the EKF
    Pi0 = scenario['Pi0']                    # confident prior, so the catch is gradual

    def make_game(route, rng):
        """
        simulate the thief along the route and its noisy observations.
        Returns None if the simulated thief does not end on the stop node of the route
        """
        turns, (i0, j0), theta0, stop_node = route
        _xx, _yy = grid.vals
        # tau = 0.1 requires eps > 2 v tau (bumps resolved by the Euler steps) and eps < dx/2
        particle = GridParticle(grid, turns, v=1.0, eps=0.25, tau=tau)
        states = particle.simulate(x0=_xx[i0, j0], y0=_yy[i0, j0], theta0=theta0, T=40, stop_node=stop_node)
        if np.linalg.norm(states[-1, :2] - [_xx[stop_node], _yy[stop_node]]) > 0.25:
            return None
        return dict(particle=particle, states=states, observations=observe(states, Q, tau, rng))

    replay_rng = np.random.default_rng() # unseeded: new thieves at each launch

    def random_game():
        while True:
            route = random_route(grid, replay_rng)
            game = make_game(route, replay_rng) if route is not None else None
            if game is not None:
                return game

    def run_ekf(game, x_prior, y_prior):
        x0 = np.array([x_prior, y_prior, -np.pi/2])  # prior heading south
        ekf = EKF(
            x0=x0, Pi0=Pi0, Q=Q, W=W, tau=tau,
            Phi=game['particle'].phi, dPhi=game['particle'].dphi, h=h, dh=dh,
            observed_trajectory=game['observations'],
            initialize_with_plus=False
        )
        ekf.run_full(len(game['observations']))
        return ekf

    first_game = make_game(FIRST_ROUTE, rng)
    if save:
        # prior far from the trajectory: (2, 7) heading south
        make_video(grid, first_game['states'], first_game['observations'], run_ekf(first_game, 2.0, 7.0), tau,
                   filename=scenario['filename'])
    else:
        # the first game uses the fixed route, the replays a new random thief each time
        games = itertools.chain([first_game], iter(random_game, None))
        interactive(grid, lambda: next(games), run_ekf, tau)
