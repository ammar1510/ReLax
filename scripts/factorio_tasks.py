"""Factorio task definitions and prompt construction for GRPO training.

Provides simple custom tasks and a loader for FLE's Lab-Play benchmark tasks.
"""

from typing import Any, Dict, List, Tuple

# ── FLE API summary included in every prompt ──────────────────────────────────

FLE_API_SUMMARY = """\
# Factorio API Reference

## Navigation
- move_to(position) — Move agent to a position
- nearest(Resource.Type) — Find nearest resource of a given type
- nearest_buildable(Prototype.Type, BuildingBox, origin) — Find buildable location

## Entity Placement & Interaction
- place_entity(Prototype.Type, position, direction=Direction.UP) — Place an entity
- place_entity_next_to(Prototype.Type, ref_position, direction, spacing=0) — Place next to reference
- rotate_entity(entity, Direction.Type) — Rotate an entity
- pickup_entity(entity) — Pick up an entity back to inventory
- get_entity(Prototype.Type, position) — Get entity reference at position

## Inventory & Crafting
- insert_item(Prototype.Type, entity, quantity) — Insert items into entity
- extract_item(Prototype.Type, entity, quantity) — Extract items from entity
- craft_item(Prototype.Type, quantity) — Craft items from inventory materials
- inspect_inventory(entity) — View entity's inventory contents

## Connections & Recipes
- connect_entities(source, target, Prototype.ConnectionType) — Connect two entities
- set_entity_recipe(entity, Prototype.RecipeType) — Set recipe on assembler

## Information
- get_entities(filter_set) — Get all entities, optionally filtered by type
- production_stats() — Get production statistics
- print(...) — Print observations to stdout

## Types
- Position(x, y) — World coordinates
- Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
- Resource.IronOre, Resource.CopperOre, Resource.Coal, Resource.Stone, Resource.Water
- Prototype.MiningDrill, Prototype.StoneFurnace, Prototype.IronChest, etc.
- Prototype.TransportBelt, Prototype.Inserter, Prototype.OffshorePump, etc.
- Prototype.AssemblingMachine1, Prototype.Boiler, Prototype.SteamEngine, etc.
- Prototype.Pipe, Prototype.SmallElectricPole, etc.
"""

# ── Simple custom tasks ───────────────────────────────────────────────────────

SIMPLE_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "mine_iron",
        "env_id": "Factorio-iron_ore_throughput_16-v0",
        "description": "Place a mining drill on the nearest iron ore deposit and connect it to a chest.",
        "difficulty": "easy",
    },
    {
        "task_id": "smelt_iron",
        "env_id": "Factorio-iron_plate_throughput_16-v0",
        "description": (
            "Set up iron smelting: place a mining drill on iron ore, "
            "connect it to a stone furnace with a transport belt, "
            "and fuel the furnace with coal."
        ),
        "difficulty": "medium",
    },
    {
        "task_id": "craft_gears",
        "env_id": "Factorio-iron_gear_wheel_throughput_16-v0",
        "description": (
            "Automate iron gear wheel production: set up iron plate smelting, "
            "then feed plates into an assembling machine configured for gear wheels."
        ),
        "difficulty": "medium",
    },
    {
        "task_id": "green_circuits",
        "env_id": "Factorio-electronic_circuit_throughput_16-v0",
        "description": (
            "Automate electronic circuit production: set up both copper wire "
            "and electronic circuit assembly lines with proper logistics."
        ),
        "difficulty": "hard",
    },
    {
        "task_id": "power_setup",
        "env_id": "Factorio-steam_engine_throughput_250-v0",
        "description": (
            "Build a power generation system: place an offshore pump near water, "
            "connect it to a boiler with pipes, fuel the boiler, "
            "and connect the boiler to a steam engine."
        ),
        "difficulty": "medium",
    },
]


def _try_load_labplay_tasks() -> List[Dict[str, Any]]:
    """Attempt to load Lab-Play benchmark tasks from FLE's task registry."""
    try:
        from fle.eval.tasks.task_definitions.task_registry import (
            list_all_tasks,
            get_task_info,
        )

        tasks = []
        for task_key in list_all_tasks():
            info = get_task_info(task_key)
            tasks.append({
                "task_id": task_key,
                "env_id": f"Factorio-{task_key}-v0",
                "description": info.get("goal", f"Complete task: {task_key}"),
                "difficulty": "labplay",
            })
        return tasks
    except ImportError:
        return []


def get_all_tasks(include_labplay: bool = True) -> List[Dict[str, Any]]:
    """Get all available training tasks.

    Args:
        include_labplay: Whether to include FLE Lab-Play benchmark tasks.

    Returns:
        List of task configuration dicts.
    """
    tasks = list(SIMPLE_TASKS)
    if include_labplay:
        labplay = _try_load_labplay_tasks()
        # Deduplicate by env_id
        existing_env_ids = {t["env_id"] for t in tasks}
        for t in labplay:
            if t["env_id"] not in existing_env_ids:
                tasks.append(t)
    return tasks


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an agent playing Factorio. You interact with the game by writing Python code \
using the Factorio API. Your code will be executed directly in the game environment.

{api_summary}

## Rules
- Write valid Python code that uses the API functions above.
- Do NOT import any modules — the API is already available in scope.
- Use print() to observe game state when needed.
- Handle errors gracefully — check positions before placing entities.
- Think step by step about what needs to be built.

## Task
{task_description}

Write Python code to accomplish this task:
"""


def build_prompt(task: Dict[str, Any]) -> str:
    """Build a text prompt for a given task.

    Args:
        task: Task configuration dict with 'description' key.

    Returns:
        Formatted prompt string.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        api_summary=FLE_API_SUMMARY,
        task_description=task["description"],
    )


def build_prompt_dataset(
    tokenizer,
    tasks: List[Dict[str, Any]],
) -> List[Tuple[List[int], Dict[str, Any]]]:
    """Build tokenized prompt dataset for GRPOTrainer.

    Args:
        tokenizer: Tokenizer with encode(text, bos, eos) method.
        tasks: List of task configuration dicts.

    Returns:
        List of (prompt_tokens, task_config) pairs compatible with
        GRPOTrainer.train(prompt_dataset=...).
    """
    dataset = []
    for task in tasks:
        prompt_text = build_prompt(task)
        tokens = tokenizer.encode(prompt_text, bos=True, eos=False)
        dataset.append((tokens, task))
    return dataset
