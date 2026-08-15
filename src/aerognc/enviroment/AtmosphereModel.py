import src.aerognc.Constants as CST
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvData:
        altitude: float
        temperature_R: float
        density_slug_ft3: float
        pressure_psf: float


class AtmosphereModel:
    Current_Conditions: EnvData = EnvData(
                                        altitude=0.0, 
                                        temperature_R=CST.TEMP_R_MSL, 
                                        density_slug_ft3=CST.DENSITY_MSL, 
                                        pressure_psf=CST.PRESSURE_MSL
                                        )

    def calculate_ISA(self, altitude: float) -> EnvData:

        Lapse_Trop = 3.57


        if altitude <= 36000:
            B = (Lapse_Trop/1000)

            T_F = CST.TEMP_F_MSL - B * altitude

            PR = (1 - (B*altitude)/CST.TEMP_R_MSL)**(CST.ONE_G/(CST.R_MSL*B))
            P_psf = PR * CST.PRESSURE_MSL

            DR = (1 - (B*altitude)/CST.TEMP_R_MSL)**((CST.ONE_G/(CST.R_MSL*B)) - 1)
            D_slug_ft3 = DR * CST.DENSITY_MSL

            self.Current_Conditions = EnvData(altitude=altitude, temperature_R=(T_F + 459.67), density_slug_ft3=D_slug_ft3, pressure_psf=P_psf)
        
        else:
             raise ValueError(f"Altitude {altitude} must be below 60,000 ft for the current model.")

        return self.Current_Conditions
        