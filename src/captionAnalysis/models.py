from django.db import models

# Create your models here.
class MyModel(models.Model):
    class Meta:
        db_table = 'JRE' # This tells Django where the SQL table is
        managed = False # Use this if table already exists
                        # and doesn't need to be managed by Django
    
    pod_id = models.IntegerField(primary_key=True)
    Title = models.CharField(max_length=150)	
    Description = models.TextField()
    Views = models.IntegerField()
    Rating = models.DecimalField(decimal_places=6,max_digits=7)
    Duration = models.IntegerField()
    Captions = models.TextField()	
    PodNum = models.IntegerField()
    TextSegments = models.TextField()	 
    CaptionWords = models.TextField()	
    Name = models.CharField(max_length=150) 
    TextIntervalDicts = models.TextField()
    TfIdfAnalysis = models.TextField()
    TFIDFvector = models.TextField()