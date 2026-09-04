// SPDX-License-Identifier: Apache-2.0
// C ABI over the adopted engine port. Python owns observation rendering and
// action parsing; this file only moves packed structs across the boundary.
#include "sim.hpp"

#include <cstdint>
#include <cstring>

namespace {

#pragma pack(push, 1)
struct PackedTile {
    std::uint8_t kind = 0;
    std::uint8_t what = 0;
    std::uint8_t flags = 0;               // 1 has_animal, 2 watered, 4 fed, 8 cared, 16 fertilizer_available
    std::int8_t consecutive_dry = 0;
    std::int8_t yield_units = 0;
    std::int8_t pending_care_bonus = 0;
    std::int16_t planted_day = 0;
    std::int32_t max_lifespan_step = -1;
    std::int16_t fertilized_until_day = -1;
};

struct PackedFarm {
    double money = 0;
    PackedTile tiles[kag::BOARD][kag::BOARD]{};
    std::int8_t pos_x[kag::MAX_UNITS]{};
    std::int8_t pos_y[kag::MAX_UNITS]{};
    std::int32_t n_units = 1;
    std::int32_t n_quadrants = 1;
    std::int32_t hires_today = 0;
    std::int16_t shed[kag::N_ITEMS]{};
    std::int16_t seeds[kag::N_CROPS]{};
    std::int16_t inv[kag::MAX_UNITS][kag::N_ITEMS]{};
    std::uint8_t inv_keys[kag::MAX_UNITS][kag::N_ITEMS]{};   // insertion order of each unit's inventory
    std::uint8_t inv_nkeys[kag::MAX_UNITS]{};
};

struct PackedState {
    std::int32_t step = 0;
    std::int32_t day = 0;
    std::int32_t hour = 0;
    std::int32_t done = 0;
    std::int32_t n_shops = 0;
    std::int32_t market_inventory[kag::N_PRODUCTS]{};
    std::int32_t market_prices[kag::N_PRODUCTS]{};
    std::uint8_t shops[kag::MAX_SHOP_INSTANCES]{};
    PackedFarm farms[2]{};
};

struct PackedAction {
    std::uint8_t unit_ops[kag::MAX_UNITS]{};
    std::uint8_t unit_args[kag::MAX_UNITS]{};
    std::int16_t unit_ns[kag::MAX_UNITS]{};
    std::int32_t n_units = 1;
    std::uint8_t order_ops[16]{};
    std::uint8_t order_items[16]{};
    std::int32_t order_ns[16]{};
    std::int32_t n_orders = 0;
};
#pragma pack(pop)

kag::Action unpack(const PackedAction& packed) {
    kag::Action action{};
    action.n_units = std::max(1, std::min(packed.n_units, kag::MAX_UNITS));
    for (int i = 0; i < action.n_units; ++i)
        action.units[i] = {packed.unit_ops[i], packed.unit_args[i], packed.unit_ns[i]};
    action.n_orders = std::max(0, std::min(packed.n_orders, 16));
    for (int i = 0; i < action.n_orders; ++i)
        action.orders[i] = {packed.order_ops[i], packed.order_items[i], packed.order_ns[i]};
    return action;
}

void pack(const kag::State& state, PackedState& out) {
    out = PackedState{};
    out.step = state.step; out.day = state.day; out.hour = state.hour; out.done = state.done;
    out.n_shops = state.n_shops;
    for (int i = 0; i < state.n_shops; ++i) out.shops[i] = state.shops[i];
    for (int i = 0; i < kag::N_PRODUCTS; ++i) {
        out.market_inventory[i] = state.market.inventory[i];
        out.market_prices[i] = state.market.prices[i];
    }
    for (int p = 0; p < 2; ++p) {
        const kag::Farm& farm = state.farms[p];
        PackedFarm& dst = out.farms[p];
        dst.money = farm.money; dst.n_units = farm.n_units; dst.n_quadrants = farm.n_quadrants;
        dst.hires_today = farm.hires_today;
        for (int u = 0; u < farm.n_units; ++u) { dst.pos_x[u] = farm.pos_x[u]; dst.pos_y[u] = farm.pos_y[u]; }
        for (int y = 0; y < kag::BOARD; ++y) for (int x = 0; x < kag::BOARD; ++x) {
            const kag::Tile& t = farm.tiles[y][x];
            PackedTile& d = dst.tiles[y][x];
            d.kind = t.kind; d.what = t.what;
            d.flags = (t.has_animal ? 1 : 0) | (t.watered_today ? 2 : 0) | (t.fed_today ? 4 : 0)
                    | (t.cared_today ? 8 : 0) | (t.fertilizer_available ? 16 : 0);
            d.consecutive_dry = t.consecutive_dry; d.yield_units = t.yield_units;
            d.pending_care_bonus = t.pending_care_bonus; d.planted_day = t.planted_day;
            d.max_lifespan_step = t.max_lifespan_step; d.fertilized_until_day = t.fertilized_until_day;
        }
        std::memcpy(dst.shed, farm.shed, sizeof dst.shed);
        std::memcpy(dst.seeds, farm.seeds, sizeof dst.seeds);
        std::memcpy(dst.inv, farm.inv, sizeof dst.inv);
        std::memcpy(dst.inv_keys, farm.inv_keys, sizeof dst.inv_keys);
        std::memcpy(dst.inv_nkeys, farm.inv_nkeys, sizeof dst.inv_nkeys);
    }
}

}  // namespace

extern "C" {

std::uint32_t kag_abi_version() { return 1; }
const char* kag_engine_version() { return kag::ENGINE_VERSION; }

void* kag_new(std::uint64_t seed, std::int32_t episode_steps) {
    kag::Config config{};
    config.seed = seed;
    config.episode_steps = episode_steps;
    return new kag::Sim(config);
}

void kag_free(void* sim) { delete static_cast<kag::Sim*>(sim); }

void kag_step(void* sim, const PackedAction* a0, const PackedAction* a1) {
    static_cast<kag::Sim*>(sim)->step(unpack(*a0), unpack(*a1));
}

void kag_export(const void* sim, PackedState* out) { pack(static_cast<const kag::Sim*>(sim)->st, *out); }

}  // extern "C"
