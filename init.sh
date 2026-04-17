git submodule sync external/fyp-power-measure
git submodule sync external/stm32ai-modelzoo
git submodule sync external/stm32ai-modelzoo-services
git submodule sync external/ultralytics
git submodule sync external/TinyissimoYOLO

# Only initialize the top-level project submodules. Some dependencies
# contain their own nested submodules that we do not want to fetch here.
git submodule update --init --depth 1 external/fyp-power-measure
git submodule update --init --depth 1 external/stm32ai-modelzoo
git submodule update --init --depth 1 external/stm32ai-modelzoo-services
git submodule update --init --depth 1 external/ultralytics
git submodule update --init --depth 1 external/TinyissimoYOLO
