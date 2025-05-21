import pygame
import os


class ControllerVisualizer:
    def __init__(self, button_names):
        pygame.init()
        self.button_names = button_names
        self.button_state = {btn: False for btn in button_names}

        self.path = os.path.join(os.path.dirname(__file__), "images")
        self.image = pygame.image.load(os.path.join(self.path, "nes.png"))

        print(f"self image: {self.image}")

        self.screen = pygame.display.set_mode((800, 800))

        # Use the already loaded image
        self.bg_image = self.image
        pygame.display.set_caption("AI Controller")
        self.font = pygame.font.SysFont("Ariel", 28)

    def update(self, pressed_buttons):
        self._draw()

    def _draw(self):
        self.screen.blit(self.bg_image, (0, 0))

        pygame.display.update()

        # Required to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
