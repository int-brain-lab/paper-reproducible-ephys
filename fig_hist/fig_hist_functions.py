
def df_to_traj_dict(df, provenance='planned'):
    traj = {
        'x': df[f'{provenance}_x'],
        'y': df[f'{provenance}_y'],
        'z': df[f'{provenance}_z'],
        'depth': df[f'{provenance}_depth'],
        'theta': df[f'{provenance}_theta'],
        'phi': df[f'{provenance}_phi']
    }
    return traj
