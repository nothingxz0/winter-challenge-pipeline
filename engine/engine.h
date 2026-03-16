/*
 * Snakebot Game Engine — Pure C API
 * Port of Rust engine (state.rs, mapgen.rs, etc.) for use with Python ctypes.
 *
 * Coordinate system: x increases rightward, y increases downward.
 * Directions: 0=NORTH(up), 1=EAST(right), 2=SOUTH(down), 3=WEST(left), 4=UNSET
 */

#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Lifecycle ---------- */
void* engine_create();
void  engine_destroy(void* handle);

/* ---------- Manual Setup (for tests) ---------- */
void  engine_set_grid(void* handle, int width, int height, const int* wall_data);
void  engine_add_bird(void* handle, int id, int owner, const int* body_xy, int body_len);
void  engine_set_bird_direction(void* handle, int bird_id, int direction); /* 0-3 or 4=unset */
void  engine_add_apple(void* handle, int ax, int ay);
void  engine_clear_apples(void* handle);
void  engine_set_turn(void* handle, int turn);
void  engine_set_losses(void* handle, int p0_losses, int p1_losses);

/* ---------- Reset for new random game ---------- */
void  engine_reset(void* handle, int seed, int league);

/* ---------- Step ----------
 * actions: flat array of [bird_id, command] pairs
 *   command: 0=N, 1=E, 2=S, 3=W  → Turn(direction)
 *            4=KEEP               → Keep (continue facing)
 * n_actions: number of pairs (array length = 2*n_actions)
 * Returns: 1 if game is terminal (turn >= max_turns or game_over), 0 otherwise
 */
int   engine_step(void* handle, const int* actions, int n_actions);

/* ---------- State Queries ---------- */
int   engine_get_turn(void* handle);
int   engine_get_width(void* handle);
int   engine_get_height(void* handle);
int   engine_is_wall(void* handle, int x, int y);
int   engine_apple_count(void* handle);
void  engine_get_apples(void* handle, int* out_xy);   /* fills [x0,y0,x1,y1,...] */
int   engine_bird_count(void* handle);
void  engine_get_bird(void* handle, int idx, int* out_id, int* out_owner,
                       int* out_alive, int* out_body_xy, int* out_len);
int   engine_body_score(void* handle, int owner);
int   engine_get_losses(void* handle, int owner);
int   engine_is_game_over(void* handle);   /* apples exhausted or player eliminated */
int   engine_is_terminal(void* handle);     /* game_over OR turn >= 200 */
int   engine_winner(void* handle);          /* -1=draw, 0=p0 wins, 1=p1 wins */

/* ---------- Legal Moves ---------- */
int   engine_legal_moves(void* handle, int bird_id, int* out_dirs);

/* ---------- Spatial Observation ----------
 * Fills a float buffer: 7 channels × height × width (caller pads to 64×64)
 * Channel layout (CHW order):
 *   0: Walls (1.0 where wall)
 *   1: Apples (1.0 where apple)
 *   2: Viewer's heads (1.0)
 *   3: Viewer's bodies (decaying 1.0→0.1 from head to tail)
 *   4: Opponent heads (1.0)
 *   5: Opponent bodies (decaying)
 *   6: Height map (y / max(height-1, 1))
 */
void  engine_get_spatial_obs(void* handle, int viewer_owner, float* out_obs);

/* ---------- Max turns configuration ---------- */
void  engine_set_max_turns(void* handle, int max_turns);

#ifdef __cplusplus
}
#endif

#endif /* ENGINE_H */
