from cobs import Model
from your_agent import Agent

Model.set_energyplus_folder("D:\\Software\\EnergyPlus\\")

if __name__ == '__main__':

    model = Model(idf_file_name="building.idf",
                  weather_file="weather.epw")
    agent = Agent()

    for _ in range(num_ep):
        state = model.reset()
        action = agent.start(state)
        while not model.is_terminate():
            state = model.step(action)
            agent.step(state)
