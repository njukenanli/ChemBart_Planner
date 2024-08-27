if __name__ == '__main__':
    from CB_Planner.board import CB_Planner
    from tasklist import tasklist
    from config import config
    planner = CB_Planner(config)
    ans = planner.plan(tasklist)
    planner.save_to_file(ans)
