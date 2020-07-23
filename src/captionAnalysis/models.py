from django.db import models

# Create your models here.
class MyModel(models.Model):
    class Meta:
        db_table = 'JRE' # This tells Django where the SQL table is
        managed = False # Use this if table already exists
                        # and doesn't need to be managed by Django
    
    pod_id = models.IntegerField(primary_key=True)
    Title = models.TextField()	
    Description = models.TextField()
    Views = models.IntegerField()
    Rating = models.DecimalField(decimal_places=6,max_digits=10)
    Duration = models.IntegerField()
    Captions = models.TextField()	
    PodNum = models.IntegerField()
    TextSegments = models.TextField()	 
    CaptionWords = models.TextField()	
    Name = models.TextField()	 
    TextIntervalDicts = models.TextField()