import mdtraj as md


def save_xtc(ref, coords, out_path):
    traj = md.Trajectory(coords, ref.topology)
    traj.superpose(ref)
    traj.save_xtc(out_path)
