import argparse
from cosolvkit.analysis import Report

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit and MD simulation afterwards.")
    parser.add_argument('-l', '--log_file', dest='log_file', required=True,
                        action='store', help='path to the log file from MD simulation')
    parser.add_argument('-traj', dest='traj_file', required=True,
                        action='store', help='path to the traj <.dcd> file from MD simulation')
    parser.add_argument('-topo', dest='top_file', required=True,
                        action='store', help='path to the topology file from MD simulation')
    parser.add_argument('-o', '--out_path', dest='out_path', required=True,
                        action='store', help='path where to store output plots')
    parser.add_argument('-c', '--cosolvents', dest='cosolvents', required=True,
                        action='store', help='path to the json file containing cosolvents used for the simulation')
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_lineparser()
    log_file = args.log_file
    traj_file = args.traj_file
    top_file = args.top_file
    out_path = args.out_path
    cosolvents_path = args.cosolvents

    report = Report(log_file, traj_file, top_file, cosolvents_path)
    report.generate_report_2(out_path=out_path)
    # report.generate_pymol_reports(report.topology, 
    #                               report.trajectory, 
    #                               density_files=["/mnt/bigdisk1/cosolvkit/manu/results/map_density_PEN.dx"], 
    #                               selection_string='', 
    #                               out_path=out_path)