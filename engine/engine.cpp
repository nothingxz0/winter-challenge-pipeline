/*
 * Snakebot Game Engine — C++ implementation
 * Faithful port of the Rust engine (state.rs, mapgen.rs, java_random.rs, map.rs).
 */

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <vector>

/* ========================================================================== */
/*  Coord / Direction / TileType                                              */
/* ========================================================================== */

struct Coord {
    int x, y;
    Coord() : x(0), y(0) {}
    Coord(int x_, int y_) : x(x_), y(y_) {}
    Coord add(int dx, int dy) const { return {x + dx, y + dy}; }
    Coord add_coord(Coord o) const { return {x + o.x, y + o.y}; }
    int manhattan_to(Coord o) const { return std::abs(x - o.x) + std::abs(y - o.y); }
    bool operator==(Coord o) const { return x == o.x && y == o.y; }
    bool operator!=(Coord o) const { return !(*this == o); }
    bool operator<(Coord o) const { return x < o.x || (x == o.x && y < o.y); }
};

enum Direction { NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3, DIR_UNSET = 4 };

static Coord dir_delta(int d) {
    switch (d) {
        case NORTH: return {0, -1};
        case EAST:  return {1,  0};
        case SOUTH: return {0,  1};
        case WEST:  return {-1, 0};
        default:    return {0,  0};
    }
}

static int dir_opposite(int d) {
    switch (d) {
        case NORTH: return SOUTH;
        case EAST:  return WEST;
        case SOUTH: return NORTH;
        case WEST:  return EAST;
        default:    return DIR_UNSET;
    }
}

static int dir_from_coord(Coord c) {
    if (c.x == 0 && c.y == -1) return NORTH;
    if (c.x == 1 && c.y == 0)  return EAST;
    if (c.x == 0 && c.y == 1)  return SOUTH;
    if (c.x == -1 && c.y == 0) return WEST;
    return DIR_UNSET;
}

enum TileType { TILE_EMPTY = 0, TILE_WALL = 1 };

/* ========================================================================== */
/*  Grid                                                                      */
/* ========================================================================== */

struct Grid {
    int width, height;
    std::vector<TileType> tiles;
    std::vector<Coord> spawns;
    std::vector<Coord> apples;

    Grid() : width(0), height(0) {}
    Grid(int w, int h) : width(w), height(h), tiles(w * h, TILE_EMPTY) {}

    bool is_valid(Coord c) const {
        return c.x >= 0 && c.x < width && c.y >= 0 && c.y < height;
    }
    int index(Coord c) const {
        if (!is_valid(c)) return -1;
        return c.y * width + c.x;
    }
    TileType get(Coord c) const {
        int idx = index(c);
        return (idx < 0) ? TILE_EMPTY : tiles[idx];
    }
    void set(Coord c, TileType t) {
        int idx = index(c);
        if (idx >= 0) tiles[idx] = t;
    }
    Coord opposite(Coord c) const { return {width - c.x - 1, c.y}; }

    std::vector<Coord> coords() const {
        std::vector<Coord> out;
        out.reserve(width * height);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                out.push_back({x, y});
        return out;
    }

    static const Coord ADJ4[4];
    static const Coord ADJ8[8];

    std::vector<Coord> neighbours4(Coord c) const {
        std::vector<Coord> out;
        for (int i = 0; i < 4; i++) {
            Coord n = c.add_coord(ADJ4[i]);
            if (is_valid(n)) out.push_back(n);
        }
        return out;
    }
    std::vector<Coord> neighbours8(Coord c) const {
        std::vector<Coord> out;
        for (int i = 0; i < 8; i++) {
            Coord n = c.add_coord(ADJ8[i]);
            if (is_valid(n)) out.push_back(n);
        }
        return out;
    }

    void sorted_unique_apples() {
        std::set<Coord> s(apples.begin(), apples.end());
        apples.assign(s.begin(), s.end());
    }

    /* Detect connected regions of empty cells */
    std::vector<std::set<Coord>> detect_air_pockets() const {
        std::set<Coord> candidates;
        for (auto& c : coords())
            if (get(c) == TILE_EMPTY) candidates.insert(c);
        return detect_regions(candidates);
    }

    /* Detect spawn islands (connected groups of spawns) */
    std::vector<std::set<Coord>> detect_spawn_islands() const {
        std::set<Coord> cand(spawns.begin(), spawns.end());
        std::vector<std::set<Coord>> result;
        std::set<Coord> seen;
        for (auto& start : spawns) {
            if (seen.count(start)) continue;
            std::set<Coord> group;
            std::deque<Coord> queue;
            queue.push_back(start);
            seen.insert(start);
            while (!queue.empty()) {
                Coord cur = queue.front(); queue.pop_front();
                group.insert(cur);
                for (auto& n : neighbours4(cur)) {
                    if (cand.count(n) && !seen.count(n)) {
                        seen.insert(n);
                        queue.push_back(n);
                    }
                }
            }
            result.push_back(group);
        }
        return result;
    }

    std::vector<Coord> detect_lowest_island() const {
        Coord start = {0, height - 1};
        if (get(start) != TILE_WALL) return {};
        std::vector<Coord> result;
        std::set<Coord> seen;
        std::deque<Coord> queue;
        queue.push_back(start);
        seen.insert(start);
        while (!queue.empty()) {
            Coord cur = queue.front(); queue.pop_front();
            result.push_back(cur);
            for (auto& n : neighbours4(cur)) {
                if (!seen.count(n) && get(n) == TILE_WALL) {
                    seen.insert(n);
                    queue.push_back(n);
                }
            }
        }
        return result;
    }

    std::vector<Coord> get_free_above(Coord c, int count) const {
        std::vector<Coord> result;
        for (int step = 1; step <= count; step++) {
            Coord above = {c.x, c.y - step};
            if (get(above) == TILE_EMPTY)
                result.push_back(above);
            else
                break;
        }
        return result;
    }

private:
    std::vector<std::set<Coord>> detect_regions(const std::set<Coord>& candidates) const {
        std::vector<std::set<Coord>> result;
        std::set<Coord> seen;
        for (auto& start : candidates) {
            if (seen.count(start)) continue;
            std::set<Coord> group;
            std::deque<Coord> queue;
            queue.push_back(start);
            seen.insert(start);
            while (!queue.empty()) {
                Coord cur = queue.front(); queue.pop_front();
                group.insert(cur);
                for (auto& n : neighbours4(cur)) {
                    if (candidates.count(n) && !seen.count(n)) {
                        seen.insert(n);
                        queue.push_back(n);
                    }
                }
            }
            result.push_back(group);
        }
        return result;
    }
};

const Coord Grid::ADJ4[4] = {{0,-1},{1,0},{0,1},{-1,0}};
const Coord Grid::ADJ8[8] = {{0,-1},{1,0},{0,1},{-1,0},{-1,-1},{1,1},{1,-1},{-1,1}};

/* ========================================================================== */
/*  JavaRandom — matches java.util.Random bit-for-bit                        */
/* ========================================================================== */

struct JavaRandom {
    static constexpr uint64_t MULTIPLIER = 0x5DEECE66DULL;
    static constexpr uint64_t ADDEND = 0xBULL;
    static constexpr uint64_t MASK = (1ULL << 48) - 1;

    uint64_t seed;

    JavaRandom() : seed(0) {}
    JavaRandom(int64_t s) {
        seed = (static_cast<uint64_t>(s) ^ MULTIPLIER) & MASK;
    }

    uint32_t next_bits(int bits) {
        seed = (seed * MULTIPLIER + ADDEND) & MASK;
        return static_cast<uint32_t>(seed >> (48 - bits));
    }

    int32_t next_int() { return static_cast<int32_t>(next_bits(32)); }

    double next_double() {
        uint64_t high = static_cast<uint64_t>(next_bits(26)) << 27;
        uint64_t low = next_bits(27);
        return static_cast<double>(high + low) / static_cast<double>(1ULL << 53);
    }

    int32_t next_int_bound(int32_t bound) {
        assert(bound > 0);
        if ((bound & -bound) == bound) {
            return static_cast<int32_t>(
                (static_cast<int64_t>(bound) * static_cast<int64_t>(next_bits(31))) >> 31);
        }
        int32_t bits, val;
        do {
            bits = static_cast<int32_t>(next_bits(31));
            val = bits % bound;
        } while (bits - val + (bound - 1) < 0);
        return val;
    }

    int32_t next_int_range(int32_t origin, int32_t bound) {
        assert(origin < bound);
        int32_t value = next_int();
        int32_t span = bound - origin;
        int32_t mask = span - 1;
        if ((span & mask) == 0) {
            return (value & mask) + origin;
        }
        if (span > 0) {
            int32_t u = static_cast<int32_t>(static_cast<uint32_t>(value) >> 1);
            for (;;) {
                int32_t v = u % span;
                if (static_cast<int32_t>(u + mask - v) >= 0) {
                    return v + origin;
                }
                u = static_cast<int32_t>(static_cast<uint32_t>(next_int()) >> 1);
            }
        }
        while (value < origin || value >= bound) {
            value = next_int();
        }
        return value;
    }

    template<typename T>
    void shuffle(std::vector<T>& v) {
        int idx = static_cast<int>(v.size());
        while (idx > 1) {
            int swap_with = next_int_bound(idx);
            std::swap(v[idx - 1], v[swap_with]);
            idx--;
        }
    }
};

/* ========================================================================== */
/*  GridMaker — map generation matching Rust mapgen.rs                        */
/* ========================================================================== */

static constexpr int MIN_GRID_HEIGHT = 10;
static constexpr int MAX_GRID_HEIGHT = 24;
static constexpr float ASPECT_RATIO = 1.8f;
static constexpr int SPAWN_HEIGHT = 3;
static constexpr int DESIRED_SPAWNS = 4;

struct GridMaker {
    JavaRandom random;
    int league_level;

    GridMaker(int64_t seed, int league) : random(seed), league_level(league) {}

    Grid make() {
        double skew;
        switch (league_level) {
            case 1: skew = 2.0; break;
            case 2: skew = 1.0; break;
            case 3: skew = 0.8; break;
            default: skew = 0.3; break;
        }

        double rand_val = random.next_double();
        int height = MIN_GRID_HEIGHT +
            static_cast<int>(std::round(
                std::pow(rand_val, skew) * (MAX_GRID_HEIGHT - MIN_GRID_HEIGHT)));
        int width = static_cast<int>(std::round(height * ASPECT_RATIO));
        if (width % 2 != 0) width++;

        Grid grid(width, height);

        double b = 5.0 + random.next_double() * 10.0;

        /* Bottom row = wall */
        for (int x = 0; x < width; x++)
            grid.set({x, height - 1}, TILE_WALL);

        /* Random walls */
        for (int y = height - 2; y >= 0; y--) {
            double y_norm = static_cast<double>(height - 1 - y) / (height - 1);
            double block_chance = 1.0 / (y_norm + 0.1) / b;
            for (int x = 0; x < width; x++) {
                if (random.next_double() < block_chance)
                    grid.set({x, y}, TILE_WALL);
            }
        }

        /* Mirror left→right */
        for (auto& c : grid.coords()) {
            Coord opp = grid.opposite(c);
            grid.set(opp, grid.get(c));
        }

        /* Fill small air pockets */
        for (auto& pocket : grid.detect_air_pockets()) {
            if (pocket.size() < 10) {
                for (auto& c : pocket)
                    grid.set(c, TILE_WALL);
            }
        }

        /* Remove walls that create tight spots (3+ wall neighbours) */
        auto all_coords = grid.coords();
        bool something_destroyed = true;
        while (something_destroyed) {
            something_destroyed = false;
            for (auto& c : all_coords) {
                if (grid.get(c) != TILE_EMPTY) continue;
                std::vector<Coord> nwalls;
                for (auto& n : grid.neighbours4(c)) {
                    if (grid.get(n) == TILE_WALL) nwalls.push_back(n);
                }
                if (nwalls.size() < 3) continue;
                std::vector<Coord> destroyable;
                for (auto& n : nwalls) {
                    if (n.y <= c.y) destroyable.push_back(n);
                }
                random.shuffle(destroyable);
                if (!destroyable.empty()) {
                    Coord target = destroyable[0];
                    grid.set(target, TILE_EMPTY);
                    grid.set(grid.opposite(target), TILE_EMPTY);
                    something_destroyed = true;
                }
            }
        }

        /* Lower the lowest island */
        auto island = grid.detect_lowest_island();
        std::set<Coord> island_set(island.begin(), island.end());
        int lower_by = 0;
        bool can_lower = true;
        while (can_lower) {
            for (int x = 0; x < width; x++) {
                Coord c = {x, height - 1 - (lower_by + 1)};
                if (!island_set.count(c)) {
                    can_lower = false;
                    break;
                }
            }
            if (can_lower) lower_by++;
        }
        if (lower_by >= 2) {
            lower_by = random.next_int_range(2, lower_by + 1);
        }

        for (auto& c : island) {
            grid.set(c, TILE_EMPTY);
            grid.set(grid.opposite(c), TILE_EMPTY);
        }
        for (auto& c : island) {
            Coord lowered = {c.x, c.y + lower_by};
            if (grid.is_valid(lowered)) {
                grid.set(lowered, TILE_WALL);
                grid.set(grid.opposite(lowered), TILE_WALL);
            }
        }

        /* Place apples */
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width / 2; x++) {
                Coord c = {x, y};
                if (grid.get(c) == TILE_EMPTY && random.next_double() < 0.025) {
                    grid.apples.push_back(c);
                    grid.apples.push_back(grid.opposite(c));
                }
            }
        }

        /* Convert isolated wall blocks to apples */
        for (auto& c : all_coords) {
            if (grid.get(c) != TILE_WALL) continue;
            int nwall8 = 0;
            for (auto& n : grid.neighbours8(c)) {
                if (grid.get(n) == TILE_WALL) nwall8++;
            }
            if (nwall8 == 0) {
                grid.set(c, TILE_EMPTY);
                Coord opp = grid.opposite(c);
                grid.set(opp, TILE_EMPTY);
                grid.apples.push_back(c);
                grid.apples.push_back(opp);
            }
        }

        /* Place spawns */
        std::vector<Coord> potential_spawns;
        for (auto& c : all_coords) {
            if (grid.get(c) == TILE_WALL &&
                static_cast<int>(grid.get_free_above(c, SPAWN_HEIGHT).size()) == SPAWN_HEIGHT) {
                potential_spawns.push_back(c);
            }
        }
        random.shuffle(potential_spawns);

        int desired = DESIRED_SPAWNS;
        if (height <= 15) desired--;
        if (height <= 10) desired--;

        while (desired > 0 && !potential_spawns.empty()) {
            Coord spawn = potential_spawns[0];
            potential_spawns.erase(potential_spawns.begin());
            auto spawn_loc = grid.get_free_above(spawn, SPAWN_HEIGHT);
            bool too_close = false;
            for (auto& c : spawn_loc) {
                if (c.x == width / 2 - 1 || c.x == width / 2) {
                    too_close = true;
                    break;
                }
                for (auto& n : grid.neighbours8(c)) {
                    for (auto& existing : grid.spawns) {
                        if (n == existing || n == grid.opposite(existing)) {
                            too_close = true;
                            break;
                        }
                    }
                    if (too_close) break;
                }
                if (too_close) break;
            }
            if (too_close) continue;

            for (auto& c : spawn_loc) {
                grid.spawns.push_back(c);
                Coord opp = grid.opposite(c);
                grid.apples.erase(
                    std::remove_if(grid.apples.begin(), grid.apples.end(),
                        [&](Coord a) { return a == c || a == opp; }),
                    grid.apples.end());
            }
            desired--;
        }

        grid.sorted_unique_apples();
        return grid;
    }
};

/* ========================================================================== */
/*  BirdState                                                                 */
/* ========================================================================== */

struct BirdState {
    int id;
    int owner;
    std::deque<Coord> body;   /* body[0] = head */
    bool alive;
    int direction;            /* NORTH..WEST or DIR_UNSET (-1 maps to DIR_UNSET) */

    BirdState() : id(0), owner(0), alive(true), direction(DIR_UNSET) {}

    Coord head() const { return body.front(); }

    int facing() const {
        if (body.size() < 2) return DIR_UNSET;
        Coord h = body[0];
        Coord neck = body[1];
        return dir_from_coord({h.x - neck.x, h.y - neck.y});
    }

    int length() const { return static_cast<int>(body.size()); }
};

/* ========================================================================== */
/*  GameState                                                                 */
/* ========================================================================== */

struct GameState {
    Grid grid;
    std::vector<BirdState> birds;
    int losses[2];
    int turn;
    int max_turns;

    GameState() : turn(0), max_turns(200) {
        losses[0] = losses[1] = 0;
    }

    void add_bird(int id, int owner, const std::vector<Coord>& body, int dir) {
        BirdState b;
        b.id = id;
        b.owner = owner;
        b.body.assign(body.begin(), body.end());
        b.alive = true;
        b.direction = dir;
        birds.push_back(b);
        /* Sort birds by id (matching Rust) */
        std::sort(birds.begin(), birds.end(),
            [](const BirdState& a, const BirdState& b) { return a.id < b.id; });
    }

    /* ---------- Scores ---------- */
    void body_scores(int out[2]) const {
        out[0] = out[1] = 0;
        for (auto& b : birds) {
            if (b.alive) out[b.owner] += b.length();
        }
    }

    void final_scores(int out[2]) const {
        body_scores(out);
        if (out[0] == out[1]) {
            out[0] -= losses[0];
            out[1] -= losses[1];
        }
    }

    /* ---------- Step ---------- */
    struct StepResult {
        bool game_over;
        int body_sc[2];
        int final_sc[2];
    };

    StepResult step(const int* p0_cmds, int p0_cnt, const int* p1_cmds, int p1_cnt) {
        turn++;
        reset_turn_state();
        apply_moves(p0_cmds, p0_cnt, p1_cmds, p1_cnt);
        apply_eats();
        apply_beheadings();
        apply_falls();

        StepResult r;
        body_scores(r.body_sc);
        final_scores(r.final_sc);
        r.game_over = is_game_over();
        return r;
    }

    bool is_game_over() const {
        if (grid.apples.empty()) return true;
        for (int owner = 0; owner <= 1; owner++) {
            bool any_alive = false;
            for (auto& b : birds) {
                if (b.owner == owner && b.alive) { any_alive = true; break; }
            }
            if (!any_alive) return true;
        }
        return false;
    }

    bool is_terminal() const {
        return is_game_over() || turn >= max_turns;
    }

    /* winner: 0=p0, 1=p1, -1=draw */
    int winner() const {
        int bs[2]; body_scores(bs);
        int body_diff = bs[0] - bs[1];
        if (body_diff > 0) return 0;
        if (body_diff < 0) return 1;
        int loss_diff = losses[1] - losses[0]; /* positive = p0 has fewer losses */
        if (loss_diff > 0) return 0;
        if (loss_diff < 0) return 1;
        return -1;
    }

    /* ---------- Legal commands for a bird ---------- */
    /* Returns directions the bird can Turn to + KEEP(4).
       A bird cannot turn to face the opposite of its current facing. */
    std::vector<int> legal_commands(int bird_id) const {
        const BirdState* bird = nullptr;
        for (auto& b : birds) {
            if (b.id == bird_id && b.alive) { bird = &b; break; }
        }
        if (!bird) return {};

        int fc = bird->facing();
        std::vector<int> cmds;
        cmds.push_back(4); /* KEEP */
        for (int d = 0; d < 4; d++) {
            if (fc != DIR_UNSET && d == dir_opposite(fc)) continue;
            if (d == fc) continue;
            cmds.push_back(d);
        }
        return cmds;
    }

    /* ---------- Spatial observation ---------- */
    void get_spatial_obs(int viewer, float* out, int W, int H) const {
        int HW = H * W;
        std::memset(out, 0, 7 * HW * sizeof(float));

        /* Ch0: Walls */
        for (int y = 0; y < grid.height && y < H; y++)
            for (int x = 0; x < grid.width && x < W; x++)
                if (grid.get({x, y}) == TILE_WALL)
                    out[0 * HW + y * W + x] = 1.0f;

        /* Ch1: Apples */
        for (auto& a : grid.apples)
            if (a.x < W && a.y < H)
                out[1 * HW + a.y * W + a.x] = 1.0f;

        /* Ch2-3: Viewer's birds, Ch4-5: Opponent's birds */
        for (auto& b : birds) {
            if (!b.alive || b.length() == 0) continue;
            bool mine = (b.owner == viewer);
            int head_ch = mine ? 2 : 4;
            int body_ch = mine ? 3 : 5;

            Coord h = b.head();
            if (h.x < W && h.y < H)
                out[head_ch * HW + h.y * W + h.x] = 1.0f;

            for (int j = 0; j < b.length(); j++) {
                Coord c = b.body[j];
                if (c.x < W && c.y < H) {
                    float decay = 1.0f - j * 0.1f;
                    if (decay < 0.1f) decay = 0.1f;
                    /* Take max in case of overlapping segments */
                    float& cell = out[body_ch * HW + c.y * W + c.x];
                    if (decay > cell) cell = decay;
                }
            }
        }

        /* Ch6: Height map */
        float hmax = std::max(grid.height - 1, 1);
        for (int y = 0; y < grid.height && y < H; y++)
            for (int x = 0; x < grid.width && x < W; x++)
                out[6 * HW + y * W + x] = static_cast<float>(y) / hmax;
    }

private:
    /* --- Step internals --- */

    void reset_turn_state() {
        for (auto& b : birds) {
            if (b.alive) b.direction = DIR_UNSET;
        }
    }

    /* Decode command arrays: each pair is [bird_id, command].
       command 0-3 = Turn(dir), 4 = Keep */
    void apply_moves(const int* p0, int p0n, const int* p1, int p1n) {
        /* Build lookup: bird_id → command */
        /* We iterate birds in order and apply the command */
        auto get_cmd = [](const int* cmds, int cnt, int bird_id) -> int {
            for (int i = 0; i < cnt; i++) {
                if (cmds[i * 2] == bird_id) return cmds[i * 2 + 1];
            }
            return 4; /* KEEP */
        };

        for (size_t idx = 0; idx < birds.size(); idx++) {
            BirdState& bird = birds[idx];
            if (!bird.alive) continue;

            int cmd = (bird.owner == 0)
                ? get_cmd(p0, p0n, bird.id)
                : get_cmd(p1, p1n, bird.id);

            if (cmd == 4) { /* KEEP */
                if (bird.direction == DIR_UNSET) {
                    bird.direction = bird.facing();
                }
            } else { /* Turn(direction) */
                int fc = bird.facing();
                if (fc != DIR_UNSET && cmd == dir_opposite(fc)) {
                    bird.direction = fc;
                } else {
                    bird.direction = cmd;
                }
            }

            int d = bird.direction;
            if (d == DIR_UNSET) d = DIR_UNSET; /* no-op, stays in place effectively */

            Coord new_head = bird.head().add_coord(dir_delta(d));

            /* Check if will eat apple */
            bool will_eat = false;
            for (auto& a : grid.apples) {
                if (a == new_head) { will_eat = true; break; }
            }

            if (!will_eat) {
                bird.body.pop_back();
            }
            bird.body.push_front(new_head);
        }
    }

    void apply_eats() {
        std::set<Coord> eaten;
        for (auto& b : birds) {
            if (!b.alive) continue;
            for (auto& a : grid.apples) {
                if (a == b.head()) { eaten.insert(a); break; }
            }
        }
        grid.apples.erase(
            std::remove_if(grid.apples.begin(), grid.apples.end(),
                [&](Coord a) { return eaten.count(a) > 0; }),
            grid.apples.end());
    }

    void apply_beheadings() {
        std::vector<size_t> alive_indices;
        for (size_t i = 0; i < birds.size(); i++)
            if (birds[i].alive) alive_indices.push_back(i);

        std::vector<size_t> to_behead;
        for (size_t idx : alive_indices) {
            Coord head = birds[idx].head();
            bool in_wall = grid.get(head) == TILE_WALL;

            /* Check if head is out of bounds (treat as wall) */
            if (!grid.is_valid(head)) in_wall = true;

            /* Check collision with any bird body */
            bool in_bird = false;
            for (auto& other : birds) {
                if (!other.alive) continue;
                for (size_t j = 0; j < other.body.size(); j++) {
                    if (other.body[j] == head) {
                        if (other.id != birds[idx].id) {
                            /* Collision with another bird */
                            in_bird = true;
                        } else if (j > 0) {
                            /* Self-collision: head overlaps own body (not position 0) */
                            in_bird = true;
                        }
                    }
                }
                if (in_bird) break;
            }

            if (in_wall || in_bird) {
                to_behead.push_back(idx);
            }
        }

        for (size_t idx : to_behead) {
            int owner = birds[idx].owner;
            if (birds[idx].length() <= 3) {
                losses[owner] += birds[idx].length();
                birds[idx].alive = false;
            } else {
                birds[idx].body.pop_front();
                losses[owner] += 1;
            }
        }
    }

    /* --- Gravity --- */
    void apply_falls() {
        bool something_fell = true;
        while (something_fell) {
            something_fell = false;
            while (apply_individual_falls()) {
                something_fell = true;
            }
            if (apply_intercoiled_falls()) {
                something_fell = true;
            }
        }
    }

    bool apply_individual_falls() {
        bool moved = false;
        std::vector<size_t> alive_idx;
        for (size_t i = 0; i < birds.size(); i++)
            if (birds[i].alive) alive_idx.push_back(i);

        for (size_t idx : alive_idx) {
            std::vector<Coord> body_vec(birds[idx].body.begin(), birds[idx].body.end());
            bool can_fall = true;
            for (auto& c : body_vec) {
                if (something_solid_under(c, body_vec)) {
                    can_fall = false;
                    break;
                }
            }
            if (can_fall) {
                moved = true;
                shift_bird_down(idx);
                /* Check if entirely below grid */
                bool all_below = true;
                for (auto& c : birds[idx].body) {
                    if (c.y < grid.height + 1) { all_below = false; break; }
                }
                if (all_below) {
                    birds[idx].alive = false;
                }
            }
        }
        return moved;
    }

    bool apply_intercoiled_falls() {
        auto groups = intercoiled_groups();
        bool moved = false;
        for (auto& group : groups) {
            /* Build meta-body of all birds in group */
            std::vector<Coord> meta;
            for (size_t idx : group) {
                for (auto& c : birds[idx].body)
                    meta.push_back(c);
            }
            bool can_fall = true;
            for (auto& c : meta) {
                if (something_solid_under(c, meta)) {
                    can_fall = false;
                    break;
                }
            }
            if (!can_fall) continue;

            moved = true;
            for (size_t idx : group) {
                shift_bird_down(idx);
                /* Note: Rust checks head().y >= height (not height+1) for intercoiled */
                if (birds[idx].head().y >= grid.height) {
                    birds[idx].alive = false;
                }
            }
        }
        return moved;
    }

    std::vector<std::vector<size_t>> intercoiled_groups() const {
        std::vector<size_t> alive_idx;
        for (size_t i = 0; i < birds.size(); i++)
            if (birds[i].alive) alive_idx.push_back(i);

        std::vector<std::vector<size_t>> groups;
        std::set<size_t> seen;

        for (size_t idx : alive_idx) {
            if (seen.count(idx)) continue;
            std::vector<size_t> group;
            std::deque<size_t> queue;
            queue.push_back(idx);
            while (!queue.empty()) {
                size_t cur = queue.front(); queue.pop_front();
                if (!seen.insert(cur).second) continue;
                group.push_back(cur);
                for (size_t other : alive_idx) {
                    if (cur == other || seen.count(other)) continue;
                    if (birds_touching(birds[cur], birds[other])) {
                        queue.push_back(other);
                    }
                }
            }
            if (group.size() > 1) {
                groups.push_back(group);
            }
        }
        return groups;
    }

    void shift_bird_down(size_t idx) {
        for (auto& c : birds[idx].body) {
            c.y += 1;
        }
    }

    bool something_solid_under(Coord c, const std::vector<Coord>& ignore_body) const {
        Coord below = c.add(0, 1);

        /* If below is in the ignore_body, not solid */
        for (auto& ig : ignore_body) {
            if (ig == below) return false;
        }

        /* Wall below */
        if (grid.get(below) == TILE_WALL) return true;

        /* Another bird body below (not in ignore list — already checked) */
        for (auto& b : birds) {
            if (!b.alive) continue;
            for (auto& seg : b.body) {
                if (seg == below) return true;
            }
        }

        /* Apple below */
        for (auto& a : grid.apples) {
            if (a == below) return true;
        }

        return false;
    }

    static bool birds_touching(const BirdState& a, const BirdState& b) {
        for (auto& ac : a.body) {
            for (auto& bc : b.body) {
                if (ac.manhattan_to(bc) == 1) return true;
            }
        }
        return false;
    }
};

/* ========================================================================== */
/*  Map generation from seed                                                  */
/* ========================================================================== */

static GameState initial_state_from_seed(int64_t seed, int league) {
    GridMaker maker(seed, league);
    Grid grid = maker.make();
    auto spawn_islands = grid.detect_spawn_islands();
    GameState state;
    state.grid = grid;
    int next_bird_id = 0;

    for (int owner = 0; owner <= 1; owner++) {
        for (auto& island : spawn_islands) {
            std::vector<Coord> body(island.begin(), island.end());
            std::sort(body.begin(), body.end());
            if (owner == 1) {
                for (auto& c : body) {
                    c = state.grid.opposite(c);
                }
            }
            state.add_bird(next_bird_id, owner, body, DIR_UNSET);
            next_bird_id++;
        }
    }
    return state;
}

/* ========================================================================== */
/*  extern "C" API implementation                                             */
/* ========================================================================== */

extern "C" {

void* engine_create() {
    return new GameState();
}

void engine_destroy(void* handle) {
    delete static_cast<GameState*>(handle);
}

void engine_set_grid(void* handle, int width, int height, const int* wall_data) {
    auto* s = static_cast<GameState*>(handle);
    s->grid = Grid(width, height);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (wall_data[y * width + x])
                s->grid.set({x, y}, TILE_WALL);
}

void engine_add_bird(void* handle, int id, int owner, const int* body_xy, int body_len) {
    auto* s = static_cast<GameState*>(handle);
    std::vector<Coord> body;
    for (int i = 0; i < body_len; i++)
        body.push_back({body_xy[i * 2], body_xy[i * 2 + 1]});
    s->add_bird(id, owner, body, DIR_UNSET);
}

void engine_set_bird_direction(void* handle, int bird_id, int direction) {
    auto* s = static_cast<GameState*>(handle);
    for (auto& b : s->birds) {
        if (b.id == bird_id) {
            b.direction = direction;
            break;
        }
    }
}

void engine_add_apple(void* handle, int ax, int ay) {
    auto* s = static_cast<GameState*>(handle);
    s->grid.apples.push_back({ax, ay});
}

void engine_clear_apples(void* handle) {
    auto* s = static_cast<GameState*>(handle);
    s->grid.apples.clear();
}

void engine_set_turn(void* handle, int turn) {
    auto* s = static_cast<GameState*>(handle);
    s->turn = turn;
}

void engine_set_losses(void* handle, int p0_losses, int p1_losses) {
    auto* s = static_cast<GameState*>(handle);
    s->losses[0] = p0_losses;
    s->losses[1] = p1_losses;
}

void engine_reset(void* handle, int seed, int league) {
    auto* s = static_cast<GameState*>(handle);
    *s = initial_state_from_seed(static_cast<int64_t>(seed), league);
}

int engine_step(void* handle, const int* actions, int n_actions) {
    auto* s = static_cast<GameState*>(handle);

    /* Split actions into p0 and p1 command arrays */
    std::vector<int> p0_cmds, p1_cmds;
    for (int i = 0; i < n_actions; i++) {
        int bird_id = actions[i * 2];
        int cmd = actions[i * 2 + 1];
        /* Find bird owner */
        int owner = -1;
        for (auto& b : s->birds) {
            if (b.id == bird_id) { owner = b.owner; break; }
        }
        if (owner == 0) {
            p0_cmds.push_back(bird_id);
            p0_cmds.push_back(cmd);
        } else if (owner == 1) {
            p1_cmds.push_back(bird_id);
            p1_cmds.push_back(cmd);
        }
    }

    s->step(
        p0_cmds.data(), static_cast<int>(p0_cmds.size() / 2),
        p1_cmds.data(), static_cast<int>(p1_cmds.size() / 2)
    );

    return s->is_terminal() ? 1 : 0;
}

int engine_get_turn(void* handle) {
    return static_cast<GameState*>(handle)->turn;
}

int engine_get_width(void* handle) {
    return static_cast<GameState*>(handle)->grid.width;
}

int engine_get_height(void* handle) {
    return static_cast<GameState*>(handle)->grid.height;
}

int engine_is_wall(void* handle, int x, int y) {
    return static_cast<GameState*>(handle)->grid.get({x, y}) == TILE_WALL ? 1 : 0;
}

int engine_apple_count(void* handle) {
    return static_cast<int>(static_cast<GameState*>(handle)->grid.apples.size());
}

void engine_get_apples(void* handle, int* out_xy) {
    auto* s = static_cast<GameState*>(handle);
    for (size_t i = 0; i < s->grid.apples.size(); i++) {
        out_xy[i * 2] = s->grid.apples[i].x;
        out_xy[i * 2 + 1] = s->grid.apples[i].y;
    }
}

int engine_bird_count(void* handle) {
    return static_cast<int>(static_cast<GameState*>(handle)->birds.size());
}

void engine_get_bird(void* handle, int idx, int* out_id, int* out_owner,
                      int* out_alive, int* out_body_xy, int* out_len) {
    auto* s = static_cast<GameState*>(handle);
    if (idx < 0 || idx >= static_cast<int>(s->birds.size())) return;
    auto& b = s->birds[idx];
    *out_id = b.id;
    *out_owner = b.owner;
    *out_alive = b.alive ? 1 : 0;
    *out_len = b.length();
    for (int i = 0; i < b.length(); i++) {
        out_body_xy[i * 2] = b.body[i].x;
        out_body_xy[i * 2 + 1] = b.body[i].y;
    }
}

int engine_body_score(void* handle, int owner) {
    auto* s = static_cast<GameState*>(handle);
    int sc[2];
    s->body_scores(sc);
    return sc[owner];
}

int engine_get_losses(void* handle, int owner) {
    return static_cast<GameState*>(handle)->losses[owner];
}

int engine_is_game_over(void* handle) {
    return static_cast<GameState*>(handle)->is_game_over() ? 1 : 0;
}

int engine_is_terminal(void* handle) {
    return static_cast<GameState*>(handle)->is_terminal() ? 1 : 0;
}

int engine_winner(void* handle) {
    return static_cast<GameState*>(handle)->winner();
}

int engine_legal_moves(void* handle, int bird_id, int* out_dirs) {
    auto* s = static_cast<GameState*>(handle);
    auto cmds = s->legal_commands(bird_id);
    int cnt = 0;
    for (int c : cmds) {
        out_dirs[cnt++] = c;
    }
    return cnt;
}

void engine_get_spatial_obs(void* handle, int viewer_owner, float* out_obs) {
    auto* s = static_cast<GameState*>(handle);
    /* Compute at grid size; caller is responsible for padding to 64x64 */
    /* Actually, we compute directly at grid size and let Python pad.
       But to simplify, we fill a W×H buffer. The Python wrapper allocates
       7*64*64 and passes it; we only fill up to grid dims. */
    s->get_spatial_obs(viewer_owner, out_obs, s->grid.width, s->grid.height);
}

void engine_set_max_turns(void* handle, int max_turns) {
    static_cast<GameState*>(handle)->max_turns = max_turns;
}

} /* extern "C" */
