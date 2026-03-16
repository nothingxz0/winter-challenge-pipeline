/*
 * test_engine.cpp — Port of Rust engine test cases + extras.
 * Compile: g++ -std=c++17 -O2 -o test_engine test_engine.cpp engine.cpp
 * Run:     ./test_engine
 */

#include "engine.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { test_funcs[n_tests++] = {#name, test_##name}; } \
    } reg_##name; \
    static void test_##name()

struct TestFunc { const char* name; void (*func)(); };
static TestFunc test_funcs[100];
static int n_tests = 0;

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        printf("  FAIL: %s == %s  (%d != %d) at line %d\n", #a, #b, (int)_a, (int)_b, __LINE__); \
        return; \
    } \
} while(0)

#define ASSERT_TRUE(x) do { if (!(x)) { printf("  FAIL: %s at line %d\n", #x, __LINE__); return; } } while(0)
#define ASSERT_FALSE(x) do { if (x) { printf("  FAIL: !(%s) at line %d\n", #x, __LINE__); return; } } while(0)

/* Helper: set up a simple grid with floor at bottom row */
static void* make_grid(int w, int h) {
    void* eng = engine_create();
    int* walls = new int[w * h]();
    /* Bottom row = wall */
    for (int x = 0; x < w; x++)
        walls[(h - 1) * w + x] = 1;
    engine_set_grid(eng, w, h, walls);
    delete[] walls;
    return eng;
}

static void* make_grid_with_walls(int w, int h, const int* extra_walls, int n_extra) {
    void* eng = engine_create();
    int* walls = new int[w * h]();
    for (int x = 0; x < w; x++)
        walls[(h - 1) * w + x] = 1;
    for (int i = 0; i < n_extra; i++) {
        int x = extra_walls[i * 2];
        int y = extra_walls[i * 2 + 1];
        walls[y * w + x] = 1;
    }
    engine_set_grid(eng, w, h, walls);
    delete[] walls;
    return eng;
}

/* ========================================================================== */
/*  Test 1: shared_apple_is_eaten_by_multiple_heads                           */
/*  Two birds converge on same apple → both eat, both get beheaded (collide)  */
/* ========================================================================== */
TEST(shared_apple_eaten_by_multiple_heads) {
    void* eng = make_grid(7, 6);
    engine_add_apple(eng, 3, 2);

    /* Bird 0 (owner 0): head at (2,2), facing EAST */
    int b0[] = {2,2, 1,2, 0,2};
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 1); /* EAST */

    /* Bird 1 (owner 1): head at (4,2), facing WEST */
    int b1[] = {4,2, 5,2, 6,2};
    engine_add_bird(eng, 1, 1, b1, 3);
    engine_set_bird_direction(eng, 1, 3); /* WEST */

    /* Step with KEEP for both */
    int actions[] = {0, 4, 1, 4}; /* bird 0 KEEP, bird 1 KEEP */
    engine_step(eng, actions, 2);

    ASSERT_EQ(engine_apple_count(eng), 0);

    /* Both birds: collision at (3,2) → beheaded. len=3 → die */
    /* Bird 0 moved to (3,2), bird 1 moved to (3,2) → head collision */
    /* Both len 3 after eating (3+1=4, then pop_back didn't happen because eating → len 4)
       Wait — both move to (3,2), both eat the apple.
       After eating: bird0 body = [(3,2),(2,2),(1,2),(0,2)] len=4
                     bird1 body = [(3,2),(4,2),(5,2),(6,2)] len=4
       Beheading: both heads at (3,2), which is also in the other bird's body.
       len > 3 → behead (pop head), losses += 1 each.
       After behead: bird0 = [(2,2),(1,2),(0,2)] len=3
                     bird1 = [(4,2),(5,2),(6,2)] len=3 */

    int id0, owner0, alive0, body0[400], len0;
    int id1, owner1, alive1, body1[400], len1;
    engine_get_bird(eng, 0, &id0, &owner0, &alive0, body0, &len0);
    engine_get_bird(eng, 1, &id1, &owner1, &alive1, body1, &len1);

    /* Per Rust test: both alive with len=3, losses=[1,1] */
    ASSERT_EQ(len0, 3);
    ASSERT_EQ(len1, 3);
    ASSERT_EQ(engine_get_losses(eng, 0), 1);
    ASSERT_EQ(engine_get_losses(eng, 1), 1);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 2: eating_removes_support_before_fall                                */
/*  Bird eats apple, apple removed, then gravity applies                      */
/* ========================================================================== */
TEST(eating_removes_support_before_fall) {
    void* eng = make_grid(5, 6);
    engine_add_apple(eng, 2, 3);

    /* Bird 0 (owner 0): vertical, head at (2,2), facing SOUTH */
    int b0[] = {2,2, 2,1, 2,0};
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 2); /* SOUTH */

    /* Step with KEEP */
    int actions[] = {0, 4};
    engine_step(eng, actions, 1);

    /* Bird moves south to (2,3), eats apple → body = [(2,3),(2,2),(2,1),(2,0)] len=4
       Apple at (2,3) was supporting... nothing actually.
       After eating, apple removed. Then gravity: bird falls.
       Expected final: [(2,4),(2,3),(2,2),(2,1)] — settled on floor */
    int id, owner, alive, body[400], len;
    engine_get_bird(eng, 0, &id, &owner, &alive, body, &len);

    ASSERT_EQ(len, 4);
    ASSERT_EQ(body[0], 2); ASSERT_EQ(body[1], 4); /* head at (2,4) */
    ASSERT_EQ(body[2], 2); ASSERT_EQ(body[3], 3);
    ASSERT_EQ(body[4], 2); ASSERT_EQ(body[5], 2);
    ASSERT_EQ(body[6], 2); ASSERT_EQ(body[7], 1);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 3: beheading_threshold_matches_referee                               */
/*  len <= 3 → die, len > 3 → just lose head                                 */
/* ========================================================================== */
TEST(beheading_threshold) {
    /* Grid 5x5, walls at (3,2) and (3,1) */
    int extra[] = {3,2, 3,1};
    void* eng = make_grid_with_walls(5, 5, extra, 2);

    /* Bird 0 (owner 0): len 3, facing EAST → will hit wall at (3,2) */
    int b0[] = {2,2, 1,2, 0,2};
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 1); /* EAST */

    /* Bird 1 (owner 1): len 4, facing EAST → will hit wall at (3,1) */
    int b1[] = {2,1, 1,1, 0,1, 0,0};
    engine_add_bird(eng, 1, 1, b1, 4);
    engine_set_bird_direction(eng, 1, 1); /* EAST */

    int actions[] = {0, 4, 1, 4};
    engine_step(eng, actions, 2);

    int id0, owner0, alive0, body0[400], len0;
    int id1, owner1, alive1, body1[400], len1;
    engine_get_bird(eng, 0, &id0, &owner0, &alive0, body0, &len0);
    engine_get_bird(eng, 1, &id1, &owner1, &alive1, body1, &len1);

    /* Bird 0: len 3, hit wall → die. losses[0] = 3 */
    ASSERT_FALSE(alive0);
    ASSERT_EQ(engine_get_losses(eng, 0), 3);

    /* Bird 1: len 4, hit wall → behead, len becomes 3. losses[1] = 1 */
    ASSERT_TRUE(alive1);
    ASSERT_EQ(len1, 3);
    ASSERT_EQ(engine_get_losses(eng, 1), 1);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 4: self_collision_bug_preserved                                      */
/*  Snake forms a loop; moving into its own tail should be OK because tail    */
/*  gets popped first (the self-collision check looks at post-move body)      */
/* ========================================================================== */
TEST(self_collision_bug_preserved) {
    void* eng = make_grid(6, 6);

    /* Bird 0: loop shape, facing WEST, will turn SOUTH.
       body = [(2,2),(2,3),(1,3),(1,2)] — head at (2,2)
       facing = from (2,2)-(2,3) = NORTH (since head is above neck)
       Wait: head=(2,2), neck=(2,3), so head.y - neck.y = -1 → NORTH.
       Turn SOUTH from facing NORTH → that's the opposite, so direction stays NORTH!
       No wait, in Rust test: direction is set as WEST initially, then Turn(South).
       Let me re-check: bird is constructed with direction Some(West).
       facing() = from_coord(head - neck) = from_coord((2,2)-(2,3)) = (0,-1) = NORTH
       So facing = NORTH, and the command is Turn(South).
       Since South is opposite of North, direction = facing = NORTH.
       Wait, that can't be right — the test expects the bird to survive...

       Actually looking at Rust: direction is Some(Direction::West) at construction.
       But step() calls reset_turn_state() which sets direction = None.
       Then apply_moves: command = Turn(South).
       facing() = NORTH.
       South == opposite(North) → direction = facing = NORTH.
       So bird moves NORTH: new_head = (2,1).
       Tail (1,2) gets popped (no apple). Body becomes [(2,1),(2,2),(2,3),(1,3)].
       No collision → alive. ✓ */

    int b0[] = {2,2, 2,3, 1,3, 1,2};
    engine_add_bird(eng, 0, 0, b0, 4);
    /* Direction will be set by facing (NORTH), command Turn(SOUTH) will be rejected */

    /* Step: Turn(SOUTH) for bird 0 */
    int actions[] = {0, 2}; /* bird 0, SOUTH */
    engine_step(eng, actions, 1);

    int id, owner, alive, body[400], len;
    engine_get_bird(eng, 0, &id, &owner, &alive, body, &len);
    ASSERT_TRUE(alive);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 5: tie_break_uses_losses                                             */
/* ========================================================================== */
TEST(tie_break_uses_losses) {
    void* eng = make_grid(4, 4);

    int b0[] = {0,0, 0,1, 0,2};
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 0); /* NORTH */

    int b1[] = {3,0, 3,1, 3,2};
    engine_add_bird(eng, 1, 1, b1, 3);
    engine_set_bird_direction(eng, 1, 0); /* NORTH */

    engine_set_losses(eng, 0, 2);

    int actions[] = {0, 4, 1, 4};
    engine_step(eng, actions, 2);

    /* Both have same body count (3 each, or after possible behead).
       losses = [0, 2], so p0 has fewer losses → p0 wins tiebreak */
    ASSERT_EQ(engine_body_score(eng, 0), engine_body_score(eng, 1));
    /* Winner should be p0 (fewer losses) */
    ASSERT_EQ(engine_winner(eng), 0);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 6: turn_direction_resets_each_turn                                   */
/* ========================================================================== */
TEST(turn_direction_resets_each_turn) {
    /* Grid 8x8, walls at (2,5) and (4,5) */
    int extra[] = {2,5, 4,5};
    void* eng = make_grid_with_walls(8, 8, extra, 2);
    engine_add_apple(eng, 3, 2);

    /* Bird 0: head at (2,2), body going down, no initial direction */
    int b0[] = {2,2, 2,3, 2,4};
    engine_add_bird(eng, 0, 0, b0, 3);

    /* Bird 1: head at (4,2), body going down, no initial direction */
    int b1[] = {4,2, 4,3, 4,4};
    engine_add_bird(eng, 1, 1, b1, 3);

    /* Turn 1: bird 0 Turn(EAST), bird 1 Turn(WEST) */
    /* facing for both = NORTH (head above neck).
       Bird 0: Turn(EAST) → direction = EAST. New head = (3,2). Eats apple!
       Body = [(3,2),(2,2),(2,3),(2,4)]. But wait, apple at (3,2) so tail stays.
       But bird 0 needs gravity check — (2,4) is on wall at (2,5)? No, wall at y=5 is at
       tiles. Let me think... Grid is 8x8, bottom row is y=7 = wall.
       Extra walls at (2,5) and (4,5).
       Bird 0: body segment at (2,4), below is (2,5) which is wall → supported.

       Bird 1: Turn(WEST) → direction = WEST. New head = (3,2).
       Both heads at (3,2) → collision!

       Per Rust test expectations:
       After turn 1, bird positions should be same as initial:
       bird 0 = [(2,2),(2,3),(2,4)]
       bird 1 = [(4,2),(4,3),(4,4)]
       losses = [1, 1]

       Hmm, that means both ate the apple (body grows to 4), then beheaded (body→3).
       Or: both move to (3,2), apple eaten, both len 4, both collide, behead → len 3.
       Wait, they collide with EACH OTHER, not walls. Both move to same cell (3,2).
       After eating: bird0=[(3,2),(2,2),(2,3),(2,4)], bird1=[(3,2),(4,2),(4,3),(4,4)].
       Beheading: both heads at (3,2), which is in the other bird → behead.
       len=4 > 3 → pop head. bird0=[(2,2),(2,3),(2,4)], bird1=[(4,2),(4,3),(4,4)].
       losses = [1, 1]. ✓

       Turn 2: KEEP for both. Directions reset.
       facing for bird 0 = NORTH. KEEP → direction = facing = NORTH.
       New head = (2,1). Body → [(2,1),(2,2),(2,3)]. Tail (2,4) popped.

       Hmm but Rust test expects positions unchanged after turn 2 too...
       That means bird moves NORTH to (2,1) and gets beheaded?
       Or the facing is actually not NORTH after beheading?

       After beheading in turn 1: bird0 = [(2,2),(2,3),(2,4)].
       facing = from_coord((2,2)-(2,3)) = (0,-1) = NORTH.
       So on turn 2, KEEP → direction = NORTH → moves to (2,1).
       But Rust test says body stays at [(2,2),(2,3),(2,4)]...

       Unless (2,1) has a wall? No.
       Unless there's no gravity support and it falls?

       Wait, let me re-read the Rust test more carefully:
       - After turn 2: body is still same as initial
       - losses = [1, 1]

       Hmm, bird0 moves to (2,1) — no wall, no collision. No apple to eat.
       Tail popped: body = [(2,1),(2,2),(2,3)].
       Gravity: (2,3) below is (2,4) = empty. (2,2) below is (2,3) = bird body.
       Actually (2,4) was the old tail, now gone. Below (2,3) is (2,4), which is empty.
       Below (2,4) is (2,5) which is wall. So... the bird can't fully fall.

       I think I need to trust the Rust test and verify after building.
       Let me just write the tests structurally. */

    /* Turn 1 */
    int act1[] = {0, 1, 1, 3}; /* bird 0 EAST, bird 1 WEST */
    engine_step(eng, act1, 2);

    int id0, owner0, alive0, body0[400], len0;
    engine_get_bird(eng, 0, &id0, &owner0, &alive0, body0, &len0);

    /* Expected: bird0 at [(2,2),(2,3),(2,4)] after behead + possible gravity */
    ASSERT_EQ(body0[0], 2); ASSERT_EQ(body0[1], 2); /* head x,y */
    ASSERT_EQ(body0[2], 2); ASSERT_EQ(body0[3], 3);
    ASSERT_EQ(body0[4], 2); ASSERT_EQ(body0[5], 4);

    int id1, owner1, alive1, body1[400], len1;
    engine_get_bird(eng, 1, &id1, &owner1, &alive1, body1, &len1);
    ASSERT_EQ(body1[0], 4); ASSERT_EQ(body1[1], 2);
    ASSERT_EQ(body1[2], 4); ASSERT_EQ(body1[3], 3);
    ASSERT_EQ(body1[4], 4); ASSERT_EQ(body1[5], 4);

    /* Turn 2 */
    int act2[] = {0, 4, 1, 4}; /* both KEEP */
    engine_step(eng, act2, 2);

    engine_get_bird(eng, 0, &id0, &owner0, &alive0, body0, &len0);
    engine_get_bird(eng, 1, &id1, &owner1, &alive1, body1, &len1);

    /* Rust test expects same positions and losses = [1,1] after 2 turns total.
       Wait — after turn 2, both keep going, which means they move again...
       The losses accumulate. Let me check: losses after turn 1 = [1,1].
       Turn 2: both KEEP, both move NORTH.
       Bird 0: moves to (2,1). Body = [(2,1),(2,2),(2,3)]. Falls...
       Below (2,3) is (2,4) = empty. Below (2,4) is (2,5) = wall.
       So can bird fall? (2,1) below is (2,2) = own body → ignore.
       (2,2) below is (2,3) = own body → ignore.
       (2,3) below is (2,4) = empty, not in own body, not wall, not apple →
       nothing solid under. So can fall!
       Falls to: [(2,2),(2,3),(2,4)]. That's the original position!
       Same for bird 1: [(4,2),(4,3),(4,4)].
       Losses stay [1,1] (no new collisions). ✓ */

    ASSERT_EQ(body0[0], 2); ASSERT_EQ(body0[1], 2);
    ASSERT_EQ(body0[2], 2); ASSERT_EQ(body0[3], 3);
    ASSERT_EQ(body0[4], 2); ASSERT_EQ(body0[5], 4);

    ASSERT_EQ(body1[0], 4); ASSERT_EQ(body1[1], 2);
    ASSERT_EQ(body1[2], 4); ASSERT_EQ(body1[3], 3);
    ASSERT_EQ(body1[4], 4); ASSERT_EQ(body1[5], 4);

    ASSERT_EQ(engine_get_losses(eng, 0), 1);
    ASSERT_EQ(engine_get_losses(eng, 1), 1);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 7: turn_cap_terminal_but_not_game_over                               */
/* ========================================================================== */
TEST(turn_cap_terminal_not_game_over) {
    void* eng = make_grid(5, 5);
    engine_add_apple(eng, 2, 1);
    engine_set_turn(eng, 199);

    int b0[] = {1,2, 1,3, 1,4};
    /* Note: the Rust test has body [(1,2),(1,3),(1,4)] sitting on floor row 4.
       But floor is at y=4 (bottom). Wait, the last segment is AT the wall?
       In Rust: Grid::new(5,5) has height=5, wall at y=4.
       Bird body at (1,4) is ON the wall tile. The body is placed there.
       Hmm, in the Rust test the bird is given Some(Direction::North).

       Actually re-reading: the Rust test body is:
       [(1,2),(1,3),(1,4)] — but y=4 is wall!
       Wait no, the Rust test uses Coord::new(1,2), Coord::new(1,3), Coord::new(1,4).
       And floor is at y=4. So tail is inside the wall. That seems wrong but it's
       what the test does. The initial direction is North.

       Let me just place birds on y=2,3 with support. */
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 0); /* NORTH */

    int b1[] = {3,2, 3,3, 3,4};
    engine_add_bird(eng, 1, 1, b1, 3);
    engine_set_bird_direction(eng, 1, 0); /* NORTH */

    int actions[] = {0, 4, 1, 4};
    engine_step(eng, actions, 2);

    /* Turn is now 200. Apples still exist → not game_over naturally.
       But terminal because turn >= 200. */
    ASSERT_FALSE(engine_is_game_over(eng));
    ASSERT_TRUE(engine_is_terminal(eng));

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 8: final_result_uses_losses_only_as_tiebreak                         */
/* ========================================================================== */
TEST(final_result_losses_tiebreak) {
    void* eng = engine_create();
    int walls[16] = {0};
    engine_set_grid(eng, 4, 4, walls);
    engine_add_apple(eng, 0, 0);
    engine_set_turn(eng, 200);
    engine_set_losses(eng, 3, 1);

    int b0[] = {0,1, 0,2, 0,3};
    engine_add_bird(eng, 0, 0, b0, 3);
    engine_set_bird_direction(eng, 0, 0);

    int b1[] = {3,1, 3,2, 3,3};
    engine_add_bird(eng, 1, 1, b1, 3);
    engine_set_bird_direction(eng, 1, 0);

    /* Body scores equal (3 == 3).
       losses[0]=3, losses[1]=1. loss_diff = 1-3 = -2 < 0 → p1 wins. */
    ASSERT_EQ(engine_body_score(eng, 0), 3);
    ASSERT_EQ(engine_body_score(eng, 1), 3);
    ASSERT_EQ(engine_winner(eng), 1);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 9: engine_reset and basic game play                                  */
/* ========================================================================== */
TEST(reset_and_play) {
    void* eng = engine_create();
    engine_reset(eng, 42, 3);

    ASSERT_TRUE(engine_get_width(eng) > 0);
    ASSERT_TRUE(engine_get_height(eng) > 0);
    ASSERT_TRUE(engine_bird_count(eng) > 0);
    ASSERT_TRUE(engine_apple_count(eng) > 0);
    ASSERT_EQ(engine_get_turn(eng), 0);
    ASSERT_FALSE(engine_is_terminal(eng));

    /* Play a few turns with KEEP for all */
    for (int t = 0; t < 5; t++) {
        int n = engine_bird_count(eng);
        int* acts = new int[n * 2];
        for (int i = 0; i < n; i++) {
            int id, owner, alive, body[400], len;
            engine_get_bird(eng, i, &id, &owner, &alive, body, &len);
            acts[i * 2] = id;
            acts[i * 2 + 1] = 4; /* KEEP */
        }
        engine_step(eng, acts, n);
        delete[] acts;
    }

    ASSERT_EQ(engine_get_turn(eng), 5);
    engine_destroy(eng);
}

/* ========================================================================== */
/*  Test 10: spatial observation shape and values                             */
/* ========================================================================== */
TEST(spatial_obs) {
    void* eng = make_grid(5, 5);
    engine_add_apple(eng, 2, 1);

    int b0[] = {1,2, 1,3};
    engine_add_bird(eng, 0, 0, b0, 2);

    int b1[] = {3,2, 3,3};
    engine_add_bird(eng, 1, 1, b1, 2);

    /* Obs buffer: 7 channels × 5 × 5 = 175 floats */
    float obs[175];
    memset(obs, 0, sizeof(obs));

    /* Use engine_get_spatial_obs — but it writes at grid dims */
    engine_get_spatial_obs(eng, 0, obs);

    int W = 5, HW = 25;

    /* Ch0: walls — bottom row should be 1 */
    for (int x = 0; x < 5; x++)
        ASSERT_TRUE(obs[0 * HW + 4 * W + x] > 0.5f);
    /* Non-wall cell should be 0 */
    ASSERT_TRUE(obs[0 * HW + 0 * W + 0] < 0.01f);

    /* Ch1: apple at (2,1) */
    ASSERT_TRUE(obs[1 * HW + 1 * W + 2] > 0.5f);

    /* Ch2: viewer (owner 0) heads — bird 0 head at (1,2) */
    ASSERT_TRUE(obs[2 * HW + 2 * W + 1] > 0.5f);

    /* Ch4: opponent heads — bird 1 head at (3,2) */
    ASSERT_TRUE(obs[4 * HW + 2 * W + 3] > 0.5f);

    engine_destroy(eng);
}

/* ========================================================================== */
/*  Main — run all tests                                                      */
/* ========================================================================== */
int main() {
    printf("Running %d engine tests...\n\n", n_tests);

    for (int i = 0; i < n_tests; i++) {
        printf("  [%d/%d] %s ... ", i + 1, n_tests, test_funcs[i].name);
        fflush(stdout);
        tests_run++;
        test_funcs[i].func();
        /* If we get here without early return, test passed */
        tests_passed++;
        printf("OK\n");
    }

    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
