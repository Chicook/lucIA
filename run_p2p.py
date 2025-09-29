import argparse
import logging
import time

from redp2p_distribuida import P2PNode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run a simple P2P node")
    parser.add_argument("--send", help="Send a text message to host (IP)")
    parser.add_argument("--text", help="Text to send", default="Hola desde lucIA P2P")
    args = parser.parse_args()

    node = P2PNode()
    node.start()

    if args.send:
        time.sleep(1.0)
        node.send_text(args.send, args.text)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()


if __name__ == "__main__":
    main()


