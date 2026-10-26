"""Validate a study's native tasks and prediction contracts without model calls.

python -m world_adapt.runner.preflight outputs/my-study/study.json --seed 3000 --out outputs/my-study/preflight.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time

from world_adapt.agent import writer
from world_adapt.environments import registry
from world_adapt.perturbations import registry as perturbations
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner.config import agent_config, build_parser, check_args, env_options, parse_args, budgets_for
from world_adapt.runner.protocol import seed_bundle
from world_adapt.runner.run import build_tasks
from world_adapt.runner.study import agents, jobs, load_study, command, pin_sources, source_drift


def _canonical(value):
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, bytes):
        return {'bytes_sha256': hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'no canonical preflight state encoding for {type(value).__name__}')


def state_hash(game, state):
    # Native PushT reset state, not its seed-based replay recipe. Coverage/status
    # derive from the exact pose and geometry; polygon-area round-off is not state identity.
    # This is not a native solver-memory dump.
    value = ({k:v for k,v in game.physical(state).items() if k not in ('coverage', 'status')}
             if game.name == 'pusht' else state)
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_job(job, agent):
    parser = build_parser('offline preflight')
    argv = command(job, str(agent.config), Path('/nonexistent/world_adapt-preflight'))[3:]
    args = parse_args(parser, argv)
    check_args(parser, args)
    spec = registry.get(job.env)
    options = env_options(args, spec)
    game = spec.make_game(options)
    levels, hashes, plan = build_tasks(args, spec, options, game, seed_bundle(job.seed))
    profiles = perturbations.profiles(job.env)
    if job.profile not in profiles:
        raise ValueError(f'{job.env}: profile is absent')
    config = agent_config(args, spec)
    fresh = spec.make_game(options)
    repeated, repeated_hashes, repeated_plan = build_tasks(args, spec, options, fresh, seed_bundle(job.seed))
    if (repeated_hashes != hashes or repeated_plan != plan
            or [(state_hash(fresh, lv.state), lv.solution) for lv in repeated]
            != [(state_hash(game, lv.state), lv.solution) for lv in levels]):
        raise ValueError(f'{job.env}: independent task generation differs')
    rows = []
    for p in plan:
        level = levels[p.level_index]
        state = level.state
        game.validate_state(state)
        budget = budgets_for(level, args)
        if budget.real_actions > args.max_turns * args.max_chunk:
            raise ValueError(f'{job.env}: turn/chunk caps cannot execute the real budget')
        sequences, cursor, replay_cursor = [], state, state
        for i in range(0, len(level.solution), args.max_chunk):
            chunk = list(level.solution[i:i+args.max_chunk])
            sequences.append((cursor, chunk))
            for a in chunk:
                if game.is_terminal(cursor) or game.is_success(cursor):
                    raise ValueError(f'{job.env}: witness has actions after terminal success/failure')
                cursor = game.step(cursor, a).state
                replay_cursor = fresh.step(replay_cursor, a).state
                game.validate_state(cursor)
                fresh.validate_state(replay_cursor)
                if (cursor != replay_cursor or game.observe(cursor) != fresh.observe(replay_cursor)
                        or game.is_success(cursor) != fresh.is_success(replay_cursor)):
                    raise ValueError(f'{job.env}: independent witness replay differs')
        witness_final = bool(level.solution) and game.is_success(cursor)
        if not witness_final:
            raise ValueError(f'{job.env}: reference witness failed')
        if len(level.solution) > budget.real_actions:
            raise ValueError(f'{job.env}: witness exceeds the real budget')
        changed = rejected = checked = visible = 0
        for start, chunk in sequences:
            identity, trace = SimulatorPredictor(game).predict(start, chunk)
            expected, cursor = [], start
            for a in chunk:
                if game.is_terminal(cursor):
                    break
                cursor = game.step(cursor, a).state
                expected.append(cursor)
            if trace.states != tuple(expected):
                raise ValueError(f'{job.env}: identity prediction differs from native execution')
            predicted, faulty = SimulatorPredictor(game, profiles[job.profile]).predict(start, chunk)
            visible += int(predicted != identity)
            changed += sum(bool(x) for x in faulty.changed)
            rejected += sum(bool(x) for x in faulty.rejected)
            checked += len(trace.states)
            # A faulty branch must not change a repeated accurate query.
            if SimulatorPredictor(game).predict(start, chunk) != (identity, trace):
                raise ValueError(f'{job.env}: predictor branch changed the real engine')
        if rejected:
            raise ValueError(f'{job.env}: {rejected} perturbations produced invalid states')
        fingerprint = level.features.get('task_fingerprint')
        if game.name == 'depot':
            from world_adapt.environments.depot.structure import task_fingerprint
            fingerprint = task_fingerprint(level.state)
        rows.append({'task_id':p.task_id,'tier':p.tier,'phase':p.phase,
                     'observation_sha256':hashes[p.level_index], 'state_sha256':state_hash(game, state),
                     **({'task_fingerprint':fingerprint} if fingerprint is not None else {}),
                     'reference_verified':bool(level.solution), 'witness_final_action_success':witness_final,
                     'budgets':asdict(budget), 'family':level.features.get('task_family', level.features.get('family')),
                     **({'initial_pose_input':level.features['initial_pose_input'],
                         'population':level.features['population']} if 'initial_pose_input' in level.features else {}),
                     'reference_length':len(level.solution),'real_budget':budget.real_actions,
                     'checked_transitions':checked,'changed_predicted_steps':changed,
                     'visible_changed_predictions':visible,'rejected_steps':rejected})
    if not profiles[job.profile].is_identity and not any(t['visible_changed_predictions'] for t in rows):
        raise ValueError(f'{job.env}: no visible perturbation effect demonstrated by offline probes')
    for engine in (game, fresh):
        close = getattr(engine, 'close', None)
        if close is not None:
            close()
    return {'environment':job.env,'seed':job.seed,'profile':job.profile,'agent':agent.name,
            'condition':job.condition,'model_calls':0,'repeat_generation_equal':True,
            'writer':writer.WRITER,'writer_input_version':writer.INPUT_VERSION,
            'writer_limits':{'tokens':config.notebook_max_tokens,'seconds':config.notebook_max_seconds,
                             'completion':config.notebook_max_completion_tokens},
            'adapter':game.spec(),'tasks':rows}


def check_structural_tasks(rows):
    """Reject structurally equivalent tasks across selected blocks and phases; pairing must be identical."""
    representatives, inventories = {}, {}
    for row in rows:
        env, seed = row['environment'], row['seed']
        for task in row['tasks']:
            fingerprint = task.get('task_fingerprint')
            if env == 'depot' and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                                   or any(c not in '0123456789abcdef' for c in fingerprint)):
                raise ValueError('depot: missing or invalid task fingerprint')
        inventory = [(t['task_id'], t['phase'], t.get('task_fingerprint')) for t in row['tasks']]
        key = (env, seed)
        if key in inventories and inventories[key] != inventory:
            raise ValueError(f'{env}: paired task inventory differs across arms or agents')
        inventories[key], representatives[key] = inventory, row
    seen = set()
    for (env, _), row in representatives.items():
        for task in row['tasks']:
            fingerprint = task.get('task_fingerprint')
            if fingerprint is None:
                continue
            key = (env, fingerprint)
            if key in seen:
                raise ValueError(f'{env}: structurally equivalent tasks across selected blocks or phases')
            seen.add(key)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--env',action='append')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    study=load_study(args.study)
    if not study['sources']:
        raise ValueError('preflight requires nonempty reviewed source pins')
    if source_drift(study):
        raise ValueError("study source pins changed; review the diff and pin the candidate before preflight")
    selected=agents(study)
    started=time.monotonic()
    rows=[]
    by_agent = {agent.name: agent for agent in selected}
    pairing = {}
    for job in jobs(study,selected,envs=args.env,seeds=args.seed):
        row=check_job(job,by_agent[job.agent]); rows.append(row)
        identity = [{k:t[k] for k in ('task_id','phase','observation_sha256','state_sha256','budgets')}
                    for t in row['tasks']]
        key = (job.env, job.seed)
        if key in pairing and pairing[key] != identity:
            raise ValueError(f'{job.env}: task/budget identity differs across enrolled arms or agents')
        pairing[key] = identity
        print(f"{job.env} seed {job.seed}: {len(row['tasks'])} tasks checked",flush=True)
    if not rows:
        parser.error('no jobs selected')
    check_structural_tasks(rows)
    for env in {r['environment'] for r in rows}:
        representatives = list({r['seed']:r for r in rows if r['environment']==env}.values())
        hashes=[t['observation_sha256'] for r in representatives for t in r['tasks']]
        if len(set(hashes))!=len(hashes):
            raise ValueError(f'{env}: duplicate observations across selected blocks')
        states = [t['state_sha256'] for r in representatives for t in r['tasks']]
        if len(set(states)) != len(states):
            raise ValueError(f'{env}: duplicate native starts across selected blocks')
    if source_drift(study):
        raise ValueError('study sources changed during preflight; no report was written')
    report={'study':study['name'],'study_sha256':hashlib.sha256(args.study.read_bytes()).hexdigest(),
            'seconds':time.monotonic()-started,'model_calls':0,'sources':pin_sources(tuple(study['sources'])),
            'checks':rows,'limitations':'Offline contracts and reference feasibility only. No claim of agent difficulty or adaptation.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
