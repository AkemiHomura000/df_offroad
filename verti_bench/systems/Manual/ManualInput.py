import evdev
from evdev import ecodes, InputDevice
import select

class ManualInput:
    def __init__(self):
        self.device = self._find_device()
        if not self.device:
            raise RuntimeError("G29 not found.")
        print(f"Found: {self.device.name} at {self.device.path}")
        self.axis_states = {}
        self._initialize_axis_states()
        self.steering_n = 0
        self.accel_n = 0
        self.brakes_n = 0
        self.direction_n = 1

    def _find_device(self):
        devices = [InputDevice(path) for path in evdev.list_devices()]
        for d in devices:
            if 'G29' in d.name:
                return d
        return None

    def _initialize_axis_states(self):
        abs_axes = self.device.capabilities().get(ecodes.EV_ABS, [])
        for code, absinfo in abs_axes:
            self.axis_states[code] = absinfo.value

    def normalize_axis(self, value, absinfo, centered=False):
        if centered:
            mid = (absinfo.max + absinfo.min) / 2
            return (value - mid) / (absinfo.max - mid)
        else:
            return (value - absinfo.min) / (absinfo.max - absinfo.min)

    def read_loop(self):
        try:
            for event in self.device.read_loop():
                if event.type == ecodes.EV_KEY:
                    # print(f"Key event: {event.code} {event.value}")
                    if event.code == 711 and event.value == 1:
                        self.direction_n *= -1
                        # print(f"Direction: {'Reverse' if self.direction_n < 0 else 'Forward'}")
                elif event.type == ecodes.EV_ABS:
                    self.axis_states[event.code] = event.value

                    get_norm = lambda c, centered=False: self.normalize_axis(
                        self.axis_states.get(c, 0), self.device.absinfo(c), centered
                    )

                    steering_n = -get_norm(ecodes.ABS_X, centered=True)
                    clutch_n = get_norm(ecodes.ABS_Y) - 1
                    brakes_n = get_norm(ecodes.ABS_RZ) - 1
                    accel_n = 1 - get_norm(ecodes.ABS_Z)
                    self.steering_n = steering_n
                    self.accel_n = accel_n
                    self.brakes_n = brakes_n
                    # print(f"\rSteering: {steering_n:.3f} | Clutch: {clutch_n:.3f} | Brakes: {brakes_n:.3f} | Accel: {accel_n:.3f}", end="")
                    
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    reader = ManualInput()
    reader.read_loop()
