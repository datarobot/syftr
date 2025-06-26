from syftr.ray.utils import ray_cache_restart, ray_init


def main():
    ray_init(force_remote=True)
    ray_cache_restart()
    print("Ray cache cleared successfully.")


if __name__ == "__main__":
    main()
