import unittest
from cognitum.model import Model
from cognitum.dataset import DatasetManager
import lmql
import subprocess

# Prompt and data adapted from the materials in https://doi.org/10.1177/20531680241231468
british_prompt = """<|start_header_id|>system<|end_header_id|>Here are some open-ended responses from the British Election Study to the question "what is the most important issue facing the country?". Please assign one of the following categories to each open ended text response, returning the original response and the most relevant label. Do not return any other text. Do not return as a list.
health: include NHS
education
election outcome
pol-neg: complaints about politics, the system, the media, corruption where no politician or party is mentioned
partisan-neg: use this code for negative statements about a party or politician
societal divides
morals: including abortion, complaints about religiosity and society's character
national identity, goals-loss: ethnonationalism
concerns about racism and discrimination
welfare
terrorism
immigration: include overpopulation. do not include refugees
asylum: asylum seekers/refugees
crime
europe: including Brexit
constitutional: UK constitutional issues except brexit, devolution and scottish independence. Include electoral system here.
international trade
devolution
scot-independence
foreign affairs: not including war
war: including Ukraine, Russia and Syria include desire for peace
defence
foreign emergency: short-term not including war
domestic emergency: short-term e.g. Flooding, storms and Grenfell Tower but not COVID
economy-general
personal finances
unemployment
taxation
debt/deficit: national/government debt not personal debt
inflation: price rises
living costs: including energy crisis
poverty
austerity: cuts and lack of spending on services
inequality
housing: including homelessness
social care: including care for elderly, adults, disabled and children
pensions/ageing
transport/infrastructure: including HS2
environment
pol values-authoritarian: pro-authoritarian responses or anti-socially liberal responses that don't fit better anywhere else: e.g. complaints about political correctness, LGBT people, wokeness etc
pol values-liberal: pro- socially liberal responses or anti-authoritarian responses that don't fit better anywhere else e.g. responses that are pro women's rights, LGBT people, free speech and other minorities or responses that are concerned about authoritarian groups
pol values-right: pro-economic right wing or anti-left wing responses that don't fit better anywhere else
pol values-left: pro-economic left wing or anti-right wing responses that don't fit better anywhere else
Referendum unspecified: where you can't tell whether response is about Brexit or indyref
Coronavirus: covid but not its economic impacts
Covid-economy
Black Lives Matter and backlash to it
other: for responses that do not fit into any other category
uncoded: for responses that cannot be reliably assigned to a category

For context, Russia invaded Ukraine prior to the fieldwork for this survey, so references to Ukraine or Russia are about war.

If multiple issues are mentioned, use the label for the first issue mentioned.

<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:
a bad economy
immergration
<|eot_id|><|start_header_id|>assistant<|end_header_id|>a bad economy|economy-general
immergration|immigration
<|eot_id|><|start_header_id|>user<|end_header_id|>Code these cases:
climate change and unemployment
<|eot_id|><|start_header_id|>assistant<|end_header_id|>climate change and unemployment|environment,unemployment
<|eot_id|><|start_header_id|>user<|end_header_id|>Please only return one code per response (if multiple match use the first issue listed). The correct response should be:

climate change and unemployment|environment
<|eot_id|><|start_header_id|>user<|end_header_id|>

Code these cases:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Here are the coded responses:

"""

class TestBritishExample(unittest.TestCase):

    def setUp(self):
        # Initialize the LMQL model
        self.lmql_model = lmql.model(
            "openai/gpt-3.5-turbo-instruct",
        )

        # Example survey data
        self.british_survey_data = [
            ("01", "COVID-19"), 
            ("02", "Covid"), 
            ("03", "Covid 19"), 
            ("04", "Coronavirus"), 
            ("05", "covid"), 
            ("06", "Economy"), 
            ("07", "Climate chang"), 
            ("08", "rejoining the eu"), 
            ("09", "economy"), 
            ("10", "Inflation"), 
            ("11", "The economy and job opportunities"), 
            ("12", "COVID-19"), 
            ("13", "Covid"), 
            ("14", "inequality"), 
            ("15", "Covid")
        ]

        # Initialize the dataset manager
        self.british_dataset = DatasetManager(self.british_survey_data)

        # Ground truth labels
        self.british_ground_truth = [
            ('01', ['coronavirus']), 
            ('02', ['coronavirus']), 
            ('03', ['coronavirus']), 
            ('04', ['coronavirus']), 
            ('05', ['coronavirus']), 
            ('06', ['economy-general']), 
            ('07', ['environment']), 
            ('08', ['europe']), 
            ('09', ['economy-general']), 
            ('10', ['inflation']), 
            ('11', ['economy-general']), 
            ('12', ['coronavirus']), 
            ('13', ['coronavirus']), 
            ('14', ['inequality']), 
            ('15', ['coronavirus'])
        ]

    def test_end_to_end(self):
        # Check dataset hash
        self.assertEqual(
            self.british_dataset.hash(), 
            "1de8d818a1d762003b17be57a6e2041160da902adad6e1446d1d2302d33af178"
        )

        # Initialize the model
        british_model = Model(model=self.lmql_model, prompt=british_prompt)

        # Generate predictions
        predictions = british_model.predict(self.british_dataset)

        # Assert predictions match ground truth
        self.assertEqual(predictions, self.british_ground_truth)

        # Evaluate predictions
        evaluation_metrics = british_model.evaluate(predictions, self.british_ground_truth)

        # Assert evaluation metrics
        self.assertEqual(evaluation_metrics.exact, 1.0)
        self.assertEqual(evaluation_metrics.partial, 0.0)
        self.assertEqual(evaluation_metrics.false_positives, 0.0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()