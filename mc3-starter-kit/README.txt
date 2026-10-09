Mini-Challenge 3 - starter kit
==============================

WHAT IS IN HERE

  mc3-corpus/            A SAMPLE corpus, the same shape as the one you are
                         graded on: 13 files across 6 subdirectories.
  sample-questions.json  The ten sample questions, each with the answer and the
                         exact citations expected. Score yourself with it.
  setup.sh               Recreates the two things a zip file cannot carry.
                         Run it once, first. Linux/macOS only.
  starter/               A working container that already implements the
                         contract. Clone it and fill in the two functions.
  selfcheck.py           Run this against your image before you submit. Same
                         checks the harness runs, in the same order.
  CONTRACT.md            The submission contract in full.

START HERE

  ./setup.sh                       # recreates the two zip-proof corpus cases
  # ... build your image, then:
  python3 selfcheck.py your-image:tag mc3-corpus

  Both selfcheck arguments are positional: your built image first, then a
  corpus directory to try it against. mc3-corpus (this sample) is fine - the
  corpus you pass is only used to prove your index pass and one query work
  end to end, so it does not need to hold the real answers.

  You still need, and this kit does not provide: Docker, an AMD GPU to build
  and test on, the mandated base image (a ~29 GiB pull, at build time, where
  the network is open), and the model weights you choose to ship.

THE GRADED CORPUS IS NOT THIS ONE

  At evaluation your container is handed a DIFFERENT corpus - larger and
  harder - at the same path, /app/corpus. It follows the same conventions,
  but the values are different. A solution that hardcodes an answer from this
  sample scores zero. The questions are hidden too: the ten here show you the
  shape, not the ones you will be asked.

THE TWO CASES A ZIP CANNOT CARRY

  setup.sh recreates both:

    - mc3-corpus/archive/ is an EMPTY directory, and some unzip tools drop it
    - mc3-corpus/vendor/internal_audit.txt must be UNREADABLE (chmod 000)

  Two warnings about that unreadable file, because each one will let a broken
  submission pass your own testing and fail ours:

    - On Windows, chmod does nothing at all. Test on Linux; Linux is what you
      are evaluated on.
    - As root, chmod 000 does not stop you reading the file either, because
      root bypasses mode bits. At evaluation we drop the DAC_OVERRIDE
      capability, which makes it genuinely raise. Check your own handling:

        docker run --cap-drop DAC_OVERRIDE ...

  A corpus walk that raises on the first unreadable file indexes nothing after
  it - and because the walk order follows the directory listing, which files
  you end up indexing then depends on filename order. That is how a submission
  scores well here and badly on the graded set.
