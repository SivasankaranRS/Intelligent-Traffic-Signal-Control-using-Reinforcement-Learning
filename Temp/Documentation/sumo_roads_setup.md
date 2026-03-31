This guide outlines the workflow for setting up a traffic simulation in SUMO using OpenStreetMap (OSM) data.

---

## 🏗️ Phase 1: Network Creation
The recommended way to build your road network is by converting real-world data.

1.  **Download:** Export a `.osm` file from [OpenStreetMap](https://www.openstreetmap.org).
2.  **Convert:** Use `netconvert` to transform the OSM file into a SUMO-compatible network.
    * *Basic command:* `netconvert --osm-files map.osm -o map.net.xml`
    * *Optimized command:* Use `--tls.guess true --junctions.join true` to fix broken junctions and estimate traffic light placement.

---

## 🚗 Phase 2: Traffic Generation
Empty roads aren't useful; you need to define vehicle movement.

1.  **Generate Trips:** Use the `randomTrips.py` tool (found in `<SUMO_HOME>/tools`) to create random origin-destination pairs.
    * `python randomTrips.py -n map.net.xml -o trips.trips.xml -e 1000`
2.  **Define Routes:** Use `duarouter` to calculate the shortest paths for those trips.
    * `duarouter -n map.net.xml -t trips.trips.xml -o routes.rou.xml`

---

## 1. `randomTrips.py` (Defining the "Who" and "Where")

Think of this as a "demand generator." Since manually typing out 1,000 vehicles in XML would be a nightmare, this Python script automates the process.

* **`-n map.net.xml`**: Tells the script to look at your specific road network so it knows which edges (roads) actually exist.
* **`-o trips.trips.xml`**: Specifies the output file. This file contains "trips," which look like this in XML: 
    `<trip id="0" depart="0.00" from="edge1" to="edge50"/>`.
* **`-e 1000`**: This stands for **End Time**. It tells the script to spread the random departures across the first 1,000 seconds of the simulation. Without this, the simulation might end the moment the first car reaches its destination.

**What it doesn't do:** It does **not** tell the car which turns to take. It only says "Car #1 starts at Road A and wants to end at Road Z."



---

## 2. `duarouter` (Defining the "How")

This is the "GPS" of the SUMO suite. It takes the "desire" created in the previous step and calculates the most logical path through the map.

* **`-n map.net.xml`**: Again, it needs the map to calculate distances and speed limits.
* **`-t trips.trips.xml`**: This is the **Input**. It reads the random start/end points you just generated.
* **`-o routes.rou.xml`**: This is the **Output**. This file is much more detailed than the trips file. 

**The Magic of DUAROUTER:**
It uses the **Dijkstra’s algorithm** (by default) to find the shortest path between the two points. The resulting `.rou.xml` file doesn't just list the start and end; it lists every single street the car must drive on:
`<route edges="edge1 edge2 edge7 edge15 edge50"/>`.

_____
## ⚙️ Phase 3: Configuration & Execution
You need to tell SUMO which files to load.

### The Config Method (Recommended)
Create a `simulation.sumocfg` file to link your network and routes:
```xml
<configuration>
    <input>
        <net-file value="map.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
</configuration>
```
**Run it:** `sumo-gui -c simulation.sumocfg`

### The Shortcut Method
Launch directly from the terminal without a config file:
`sumo-gui -n map.net.xml -r routes.rou.xml`

---

## Increasing the traffic
### 1. Increase the Frequency (The `--period` flag)
The easiest way to add more cars is the `--period` flag in `randomTrips.py`. This tells SUMO how many seconds to wait between spawning each new vehicle.

* **Lower number = More traffic.**
* **Example:** `python randomTrips.py -n map.net.xml -o trips.trips.xml -e 3600 --period 0.2`

> **What this does:** Spawns a new car every 0.2 seconds (5 cars per second). Over an hour (`-e 3600`), this will put **18,000 cars** on your map.

---

### 2. Increase the Duration (The `-e` flag)
If your simulation ends too quickly or cars stop appearing, your "End Time" is too low. For a huge map, cars might take 500 seconds just to cross it. If you stop spawning them at 1,000 seconds, the map will be empty by second 1,500.

* **Fix:** Set `-e` to a much higher number, like `3600` (1 hour) or `86400` (24 hours).
* **Command:** `python randomTrips.py -n map.net.xml -o trips.trips.xml -e 10000`

---

### 3. Use "Fringe" Starting Points (The `--fringe-factor` flag)
By default, `randomTrips.py` picks any random street. On a huge map, many cars might start and end on tiny side streets, so you never see them on the main highways where your RL traffic lights are.

* **The Fix:** Use `--fringe-factor`. This makes it much more likely for cars to spawn at the **edges** of your map and drive across the whole thing.
* **Example:** `python randomTrips.py -n map.net.xml -o trips.trips.xml -e 3600 --fringe-factor 10`

---

### 🚀 Recommended "Heavy Traffic" Command
If you want a busy map for your RL project, run this:

```bash
python randomTrips.py -n map.net.xml -o trips.trips.xml -e 3600 --period 0.1 --fringe-factor 10 --junctions.join
```

Then, don't forget to re-run the router:
```bash
duarouter -n map.net.xml -t trips.trips.xml -o routes.rou.xml --ignore-errors
```
*(Note: I added `--ignore-errors` because on huge maps, some random start/end points might not have a valid path between them. This flag prevents the process from crashing.)*
