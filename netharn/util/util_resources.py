def ensure_ulimit():
    # NOTE: It is important to have a high enought ulimit for DataParallel
    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if rlimit[0] <= 8192:
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
    except Exception:
        print('Unable to fix ulimit. Ensure manually')
        raise
