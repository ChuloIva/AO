The world model and the "patient' in the scenario are the same instance, lets brake this down into 2 instances, and that for each response, there is a world response, noting all the things and everything as the game unfolds (returning results of things) and then there is patient which is an instance in itself, we don't share the world context with the patient instance, and we share the world context with the doctor instance. Also let me decide the patient and default world models in @config.yaml 

Also I would need to add to UI so that when capturing activations that I can select multiple points, so that I can capture at 25 50 and 75 and have all of those activations up for interpreting 

also I would want to have a button to stop the simulation (scenario) and so that we can move to analysis phase, so it doesn't need to "end" in sense, we can at any point move to analysis phase

also there is a thing in @clickable_components.md The grid of buttons currently is not so good with large conversations, so can we also implement that